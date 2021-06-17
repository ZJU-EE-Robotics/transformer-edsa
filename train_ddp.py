import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist

import hparams as hparams
from model import Transformer
from utils_data import TextMelLoader, TextMelCollate
from loss_function import TransformerLoss
from logger import TransformerLogger
from utils_public import parse_batch


def lr_schdule(optimizer, iteration):
    _iteration = iteration + 1
    lr = hparams.learning_rate * min(
        _iteration * hparams.warmup_step ** -1.5, _iteration ** -0.5
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer, lr


def load_model(hparams, gpu_rank):
    model = Transformer(hparams).cuda(gpu_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_rank])
    return model


def prepare_directories_and_logger(output_directory, log_directory, gpu_rank):
    if gpu_rank == 0:
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
        os.makedirs(log_directory, exist_ok=True)
        os.chmod(log_directory, 0o775)
        logger = TransformerLogger(log_directory)
    else:
        logger = None
    return logger


def prepare_dataloaders(hparams, processing_rank, world_size):
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=world_size, rank=processing_rank
    )

    train_loader = DataLoader(
        trainset,
        num_workers=2,
        shuffle=False,
        batch_size=hparams.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    valid_loader = DataLoader(
        valset,
        num_workers=2,
        shuffle=True,
        batch_size=hparams.batch_size,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    learning_rate = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]
    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath
        )
    )
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        filepath,
    )


def validate(model, criterion, valid_loader):

    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sum = 0.0
        val_mel_l1 = 0.0
        val_mel_l2 = 0.0
        val_gate = 0.0
        val_guide = 0.0
        for i, batch in enumerate(valid_loader):
            x, y = parse_batch(batch)
            y_pred = model(x)
            loss, loss_meta = criterion(y_pred, y)
            val_sum += loss.item()
            val_mel_l1 += loss_meta[0].item()
            val_mel_l2 += loss_meta[1].item()
            val_gate += loss_meta[2].item()
            val_guide += loss_meta[3].item()
        val_sum = val_sum / (i + 1)
        val_mel_l1 = val_mel_l1 / (i + 1)
        val_mel_l2 = val_mel_l2 / (i + 1)
        val_gate = val_gate / (i + 1)
        val_guide = val_guide / (i + 1)

    model.train()
    return (val_sum, val_mel_l1, val_mel_l2, val_gate, val_guide)


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    valid_loader,
    iteration,
    epoch_offset,
    logger,
    output_directory,
    gpu_rank,
):

    model.train()
    model.zero_grad()
    accum_loss = 0.0
    accum_loss_mel_l1 = 0.0
    accum_loss_mel_l2 = 0.0
    accum_loss_gate = 0.0
    accum_loss_guide = 0.0
    accum_dur = 0.0

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for batch in train_loader:
            accum_iter = iteration // hparams.accum_size
            optimizer, learning_rate = lr_schdule(optimizer, accum_iter)

            # the start time
            x, y = parse_batch(batch, gpu_rank)
            start = time.perf_counter()
            y_pred = model(x)
            loss, loss_meta = criterion(y_pred, y)
            loss = loss / hparams.accum_size
            loss_meta = [item / hparams.accum_size for item in loss_meta]
            loss.backward()

            accum_loss += loss.item()
            accum_loss_mel_l1 += loss_meta[0].item()
            accum_loss_mel_l2 += loss_meta[1].item()
            accum_loss_gate += loss_meta[2].item()
            accum_loss_guide += loss_meta[3].item()

            train_loss = (
                accum_loss,
                accum_loss_mel_l1,
                accum_loss_mel_l2,
                accum_loss_gate,
                accum_loss_guide,
            )

            # the end time
            duration = time.perf_counter() - start
            accum_dur += duration

            # accumulate gradients
            if (iteration + 1) % hparams.accum_size == 0:
                # clip abnormal gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh
                )
                optimizer.step()
                model.zero_grad()

                if gpu_rank == 0:
                    print(
                        "Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                            accum_iter, accum_loss, grad_norm, accum_dur
                        )
                    )
                    logger.log_training(
                        train_loss, grad_norm, learning_rate, accum_dur, accum_iter,
                    )
                accum_loss = 0.0
                accum_loss_mel_l1 = 0.0
                accum_loss_mel_l2 = 0.0
                accum_loss_gate = 0.0
                accum_loss_guide = 0.0
                accum_dur = 0.0

                # validate the model
                if accum_iter % hparams.iters_per_checkpoint == 0 and gpu_rank == 0:
                    val_loss = validate(model.module, criterion, valid_loader)
                    print("Validation loss {}: {:9f}  ".format(accum_iter, val_loss[0]))
                    logger.log_validation(val_loss, model.module, y, y_pred, accum_iter)
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(accum_iter)
                    )
                    save_checkpoint(
                        model.module,
                        optimizer,
                        learning_rate,
                        iteration,
                        checkpoint_path,
                    )
            iteration += 1


def main(gpu_rank, args):
    output_directory = args.output_directory
    log_directory = args.log_directory
    checkpoint_path = args.checkpoint_path
    processing_rank = args.nr * args.gpus + gpu_rank

    # for distributed dataparallel
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=processing_rank,
    )

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    model = load_model(hparams, gpu_rank)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        betas=(hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps,
        weight_decay=hparams.weight_decay,
    )
    criterion = TransformerLoss(guide_attn=True).cuda(gpu_rank)
    logger = prepare_directories_and_logger(output_directory, log_directory, gpu_rank)
    train_loader, valid_loader = prepare_dataloaders(
        hparams, processing_rank, args.world_size
    )

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        model.module, optimizer, _, iteration = load_checkpoint(
            checkpoint_path, model.module, optimizer
        )
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))

    train(
        model,
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        iteration,
        epoch_offset,
        logger,
        output_directory,
        gpu_rank,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default="/home/server/disk1/checkpoints/transformer_tts_frt/exp_ddp",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "-l",
        "--log_directory",
        type=str,
        default="/home/server/disk1/checkpoints/transformer_tts_frt/log/exp_ddp",
        help="directory to save tensorboard logs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "-n", "--nodes", type=int, default=1, help="number of the nodes used"
    )
    parser.add_argument(
        "-g", "--gpus", type=int, default=1, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr", "--nr", type=int, default=0, help="ranking within the nodes"
    )
    args = parser.parse_args()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    # for distributed dataparallel
    args.world_size = args.gpus * args.nodes
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8765"
    mp.spawn(main, nprocs=args.gpus, args=(args,))
