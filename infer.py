import os
import argparse
import matplotlib.pylab as plt
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import hparams
from model_edsa import Transformer
from utils_data import TextMelLoader, TextMelCollate
from utils_public import parse_batch


def plot_data(data, index, path, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect="auto", origin="bottom", interpolation="none")
    file = os.path.join(path, str(index) + "-mel.png")
    plt.savefig(file)
    plt.close()


def denormalize_feats(feat, cmvn_path):
    feat = feat.detach().cpu().numpy()
    cmvn = np.load(os.path.join(cmvn_path, "cmvn.npy"))
    mean = cmvn[:, 0:1]
    std = cmvn[:, 1:]
    feat = (feat * std) + mean
    feat = torch.from_numpy(feat)
    return feat


def parse_attn(alignments):
    # * is needed to convert the list of alignments to iterable arguments.
    alignments = zip(*alignments)
    dec_enc_attn_list = []
    for idx, alignment in enumerate(alignments):
        alignment = torch.cat(alignment, dim=-2)
        dec_enc_attn_list.append(alignment)
    return dec_enc_attn_list


def plot_attn(enc_attn_list, dec_attn_list, dec_enc_attn_list, index, path):
    if type(dec_attn_list) is str:
        if dec_attn_list == "mode_dp":
            dec_enc_attn_list = parse_attn(dec_enc_attn_list)

    # enc_attn_list[0]'s shape (b, h, lq, lk)
    layer_num = len(enc_attn_list)
    heads_num = enc_attn_list[0].size(1)

    # encoder attn image
    fig, axes = plt.subplots(layer_num, heads_num, figsize=(16, 12))
    for i in range(layer_num):
        for j in range(heads_num):
            # (lk, lq)
            enc_attn = enc_attn_list[i][0, j].detach().cpu().numpy().T
            axes[i][j].imshow(
                enc_attn, aspect="auto", origin="bottom", interpolation="none"
            )
    file = os.path.join(path, str(index) + "-enc-attn.png")
    plt.savefig(file)
    plt.close()

    # decoder attn image
    """
    fig, axes = plt.subplots(layer_num, heads_num, figsize=(16, 12))
    for i in range(layer_num):
        for j in range(heads_num):
            # (lk, lq)
            enc_attn = dec_attn_list[i][0, j].detach().cpu().numpy().T
            axes[i][j].imshow(
                enc_attn, aspect="auto", origin="bottom", interpolation="none"
            )
    file = os.path.join(path, str(index) + "-dec-attn.png")
    plt.savefig(file)
    plt.close()
    """

    # decoder-encoder image
    fig, axes = plt.subplots(layer_num, heads_num, figsize=(16, 12))
    for i in range(layer_num):
        for j in range(heads_num):
            # (lk, lq)
            enc_attn = dec_enc_attn_list[i][0, j].detach().cpu().numpy().T
            axes[i][j].imshow(
                enc_attn, aspect="auto", origin="bottom", interpolation="none"
            )
    file = os.path.join(path, str(index) + "-dec-enc-attn.png")
    plt.savefig(file)
    plt.close()


def load_avg_checkpoint(checkpoint_path):
    checkpoint_restore = torch.load(checkpoint_path[0])["state_dict"]
    for idx in range(1, len(checkpoint_path)):
        checkpoint_add = torch.load(checkpoint_path[idx])["state_dict"]
        for key in checkpoint_restore:
            checkpoint_restore[key] = checkpoint_restore[key] + checkpoint_add[key]

    for key in checkpoint_restore:
        if key.split(".")[-1] == "num_batches_tracked":
            checkpoint_restore[key] = checkpoint_restore[key] // (len(checkpoint_path))
        else:
            checkpoint_restore[key] = checkpoint_restore[key] / (len(checkpoint_path))
    return checkpoint_restore


def main(args, hparams):

    # prepare data
    testset = TextMelLoader(hparams.test_files, hparams, shuffle=False)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    test_loader = DataLoader(
        testset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # prepare model
    model = Transformer(hparams).cuda("cuda:0")
    checkpoint_restore = load_avg_checkpoint(args.checkpoint_path)
    model.load_state_dict(checkpoint_restore)
    model.eval()
    print("# total parameters:", sum(p.numel() for p in model.parameters()))

    # infer
    duration_add = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            x, y = parse_batch(batch)

            # the start time
            start = time.perf_counter()
            (
                mel_output,
                mel_output_postnet,
                _,
                enc_attn_list,
                dec_attn_list,
                dec_enc_attn_list,
            ) = model.inference_dp(x, force_layer_list=[5])

            # the end time
            duration = time.perf_counter() - start
            duration_add += duration

            # denormalize the feats and save the mels and attention plots
            mel_predict = mel_output_postnet[0]
            mel_denorm = denormalize_feats(mel_predict, hparams.dump)
            mel_path = os.path.join(args.output_infer, "{:0>3d}".format(i) + ".pt")
            torch.save(mel_denorm, mel_path)

            plot_data(
                (
                    mel_output.detach().cpu().numpy()[0],
                    mel_output_postnet.detach().cpu().numpy()[0],
                    mel_denorm.numpy(),
                ),
                i,
                args.output_infer,
            )

            plot_attn(
                enc_attn_list, dec_attn_list, dec_enc_attn_list, i, args.output_infer,
            )

        duration_avg = duration_add / (i + 1)
        print("The average inference time is: %f" % duration_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_infer",
        type=str,
        default="output_infer",
        help="directory to save infer outputs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=list,
        default=[
            "/home/server/disk1/checkpoints/transformer_tts_frt/exp_ddp_amp/checkpoint_154000",
        ],
        required=False,
        help="checkpoint path for infer model",
    )
    args = parser.parse_args()
    os.makedirs(args.output_infer, exist_ok=True)
    assert args.checkpoint_path is not None

    main(args, hparams)
    print("finished")
