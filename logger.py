import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from utils_plotting import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from utils_plotting import plot_gate_outputs_to_numpy


class TransformerLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TransformerLogger, self).__init__(logdir)

    def log_training(self, loss, grad_norm, lr, duration, iteration):
        self.add_scalar("training/loss", loss[0], iteration)
        self.add_scalar("training/loss_mel_l1", loss[1], iteration)
        self.add_scalar("training/loss_mel_l2", loss[2], iteration)
        self.add_scalar("training/loss_gate", loss[3], iteration)
        self.add_scalar("training/loss_guide", loss[4], iteration)
        self.add_scalar("training/grad_norm", grad_norm, iteration)
        self.add_scalar("training/learning_rate", lr, iteration)
        self.add_scalar("training/duration", duration, iteration)

    def log_validation(self, loss, model, y, y_pred, iteration):
        self.add_scalar("validation/loss", loss[0], iteration)
        self.add_scalar("validation/loss_mel_l1", loss[1], iteration)
        self.add_scalar("validation/loss_mel_l2", loss[2], iteration)
        self.add_scalar("validation/loss_gate", loss[3], iteration)
        self.add_scalar("validation/loss_guide", loss[4], iteration)
        self.add_scalars(
            "validation/alpha",
            {
                "alpha_txt": model.encoder_prenet.position.alpha.item(),
                "alpha_mel": model.decoder_prenet.position.alpha.item(),
            },
            iteration,
        )

        (
            _,
            mel_output_postnet,
            gate_output,
            enc_attn_list,
            dec_attn_list,
            dec_enc_attn_list,
        ) = y_pred[0]
        mel_target, gate_target = y

        # plot distribution of parameters
        # for tag, value in model.named_parameters():
        #     tag = tag.replace(".", "/")
        #     self.add_histogram(tag, value.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, mel_target.size(0) - 1)

        self.plot_multihead_attention("encoder/", idx, enc_attn_list, iteration)
        # self.plot_multihead_attention("decoder/", idx, dec_attn_list, iteration)
        self.plot_multihead_attention(
            "decoder_encoder/", idx, dec_enc_attn_list, iteration
        )

        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(
                mel_target[idx].detach().cpu().numpy().astype(np.float32)
            ),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(
                mel_output_postnet[idx].detach().cpu().numpy().astype(np.float32)
            ),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_target[idx].detach().cpu().numpy(),
                torch.sigmoid(gate_output[idx]).detach().cpu().numpy(),
            ),
            iteration,
            dataformats="HWC",
        )

    def plot_multihead_attention(self, info, index, attention_list, iteration):

        # attention_list, [(N, head_num, lq, lk), ..., ]
        for i in range(len(attention_list)):

            # (head_num, lq, lk)
            multihead_attn = attention_list[i][index]
            head_num = multihead_attn.size(0)
            for j in range(head_num):

                # (lq, lk)
                alignment = multihead_attn[j]
                self.add_image(
                    info + "layer_" + str(i) + "_head_" + str(j),
                    plot_alignment_to_numpy(alignment.detach().cpu().numpy().T),
                    iteration,
                    dataformats="HWC",
                )
