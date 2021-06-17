import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_output_mask(out_flag):

    # (N, L)
    output_mask = out_flag.eq(0)
    return output_mask


def get_attn_mask(seq_flag):
    """
    generate the mask for attention, 0 for unmask and 1 for mask
    Args:
        seq_flag - (N, L).

    Return:
        qmask - (N, L, 1) for query side mask.
        kmask - (N, 1, L) for key and value side mask.
    """
    mask = seq_flag.eq(0)
    qmask = mask.unsqueeze(-1)
    kmask = mask.unsqueeze(1)
    return qmask, kmask


def get_causal_mask(seq):
    """
    For masking out the subsequent info.
    Args:
        seq (N, _, L)

    Return:
        mask (1, L, L), 0 for unmask and 1 for mask
    """
    causal_mask = (
        torch.triu(
            torch.ones((1, seq.size(-1), seq.size(-1)), device=seq.device), diagonal=1,
        )
    ).bool()
    return causal_mask


class PositionalEncoding(nn.Module):
    def __init__(self, n_position, d_hid):
        super(PositionalEncoding, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.alpha * self.pos_table[:, : x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, qmask=None, kmask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # masking for key side, set -1e9 to minimize the cor attn score
        if kmask is not None:
            attn = attn.masked_fill(kmask, -10000.0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        # masking for query side, ignore the all padding postions for clean plot
        if qmask is not None:
            attn = attn.masked_fill(qmask, 0.0)

        # output (N, H, Lq, Dv), attn (N, H, Lq, Lk)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, qmask=None, kmask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if qmask is not None:
            qmask = qmask.unsqueeze(1)  # For head axis broadcasting.

        if kmask is not None:
            kmask = kmask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, qmask=qmask, kmask=kmask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads
        # together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """
    Composed with two sub-layers, i.e. Multi-Head Attention (encoder) and FFN.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, input, enc_qmask=None, enc_kmask=None):
        output, enc_attn = self.multi_head_attn(
            input, input, input, qmask=enc_qmask, kmask=enc_kmask
        )
        output = self.ffn(output)
        return output, enc_attn


class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    hparams.d_model,
                    hparams.d_inner,
                    hparams.n_head,
                    hparams.d_model // hparams.n_head,
                    hparams.d_model // hparams.n_head,
                    dropout=0.1,
                )
                for _ in range(hparams.n_layers)
            ]
        )

    def forward(self, input, enc_qmask=None, enc_kmask=None, return_attns=False):
        enc_attn_list = []

        # positional dropout
        output = F.dropout(input, p=0.1, training=self.training)

        for layer in self.layer_stack:
            output, head_attn = layer(output, enc_qmask=enc_qmask, enc_kmask=enc_kmask)
            enc_attn_list += [head_attn] if return_attns else []

        if return_attns:
            return output, enc_attn_list
        return output


class EncoderPrenet(nn.Module):
    def __init__(self, hparams):
        super(EncoderPrenet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    hparams.d_embed,
                    hparams.eprenet_chans,
                    kernel_size=hparams.eprenet_kernel_size,
                    stride=1,
                    padding=int((hparams.eprenet_kernel_size - 1) / 2),
                ),
                nn.BatchNorm1d(hparams.eprenet_chans),
            )
        )

        for _ in range(hparams.eprenet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        hparams.eprenet_chans,
                        hparams.eprenet_chans,
                        kernel_size=hparams.eprenet_kernel_size,
                        stride=1,
                        padding=int((hparams.eprenet_kernel_size - 1) / 2),
                    ),
                    nn.BatchNorm1d(hparams.eprenet_chans),
                )
            )

        self.project = nn.Linear(hparams.eprenet_chans, hparams.d_model)
        self.txt_embed = nn.Embedding(hparams.n_symbols, hparams.d_embed, padding_idx=0,)
        self.position = PositionalEncoding(hparams.n_position, hparams.d_model)

    def forward(self, txt_seq):

        # (N, C, L)
        x = self.txt_embed(txt_seq).transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), p=0.5, training=self.training)

        # (N, L, C)
        x = self.project(x.transpose(1, 2))
        x = self.position(x)
        return x


class DecoderLayer(nn.Module):
    """
    Composed with three sub-layers, i.e. Masked Multi-Head Attention (decoder),
    Multi-Head Attention (encoder-decoder) and FFN.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_head_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.multi_head_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
        self, input, enc_output, dec_qmask=None, dec_kmask=None, dec_enc_kmask=None,
    ):
        output, dec_attn = self.masked_head_attn(
            input, input, input, qmask=dec_qmask, kmask=dec_kmask
        )
        output, dec_enc_attn = self.multi_head_attn(
            output, enc_output, enc_output, qmask=dec_qmask, kmask=dec_enc_kmask,
        )
        output = self.ffn(output)
        return output, dec_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    hparams.d_model,
                    hparams.d_inner,
                    hparams.n_head,
                    hparams.d_model // hparams.n_head,
                    hparams.d_model // hparams.n_head,
                    dropout=0.1,
                )
                for _ in range(hparams.n_layers)
            ]
        )

    def forward(
        self, input, dec_qmask, dec_kmask, enc_output, dec_enc_kmask, return_attns=False,
    ):
        dec_attn_list, dec_enc_attn_list = [], []

        # positional dropout
        output = F.dropout(input, p=0.1, training=self.training)

        for layer in self.layer_stack:
            output, dec_attn, dec_enc_attn = layer(
                output,
                enc_output,
                dec_qmask=dec_qmask,
                dec_kmask=dec_kmask,
                dec_enc_kmask=dec_enc_kmask,
            )
            dec_attn_list += [dec_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return output, dec_attn_list, dec_enc_attn_list
        return output


class DecoderPrenet(nn.Module):
    def __init__(self, d_input, d_prenet, d_model, n_position):
        super(DecoderPrenet, self).__init__()
        sizes = [d_prenet, d_prenet]
        in_sizes = [d_input] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

        self.project = nn.Linear(sizes[-1], d_model)
        self.position = PositionalEncoding(n_position, d_model)

    def forward(self, mel_seq):
        # (N, L, C)
        x = mel_seq.transpose(1, 2)
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=0.5, training=True)

        x = self.project(x)
        x = self.position(x)
        return x


class Postnet(nn.Module):
    """
    Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    hparams.d_mel,
                    hparams.dpostnet_chans,
                    kernel_size=hparams.dpostnet_kernel_size,
                    stride=1,
                    padding=int((hparams.dpostnet_kernel_size - 1) / 2),
                ),
                nn.BatchNorm1d(hparams.dpostnet_chans),
            )
        )

        for i in range(1, hparams.dpostnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        hparams.dpostnet_chans,
                        hparams.dpostnet_chans,
                        kernel_size=hparams.dpostnet_kernel_size,
                        stride=1,
                        padding=int((hparams.dpostnet_kernel_size - 1) / 2),
                    ),
                    nn.BatchNorm1d(hparams.dpostnet_chans),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    hparams.dpostnet_chans,
                    hparams.d_mel,
                    kernel_size=hparams.dpostnet_kernel_size,
                    stride=1,
                    padding=int((hparams.dpostnet_kernel_size - 1) / 2),
                ),
                nn.BatchNorm1d(hparams.d_mel),
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(
                torch.tanh(self.convolutions[i](x)), p=0.5, training=self.training,
            )
        x = F.dropout(self.convolutions[-1](x), p=0.5, training=self.training)
        return x


class Transformer(nn.Module):
    def __init__(self, hparams):
        super(Transformer, self).__init__()
        self.encoder_prenet = EncoderPrenet(hparams)
        self.encoder = Encoder(hparams)
        self.decoder_prenet = DecoderPrenet(
            hparams.d_mel * hparams.n_frames_per_step,
            hparams.dprenet_size,
            hparams.d_model,
            hparams.n_position,
        )
        self.decoder = Decoder(hparams)
        self.d_mel = hparams.d_mel
        self.n_frames_per_step = hparams.n_frames_per_step
        self.stop_threshold = hparams.stop_threshold
        self.max_decoder_steps = hparams.max_decoder_steps
        self.infer_trim = hparams.infer_trim

        self.mel_linear = nn.Linear(
            hparams.d_model, hparams.d_mel * hparams.n_frames_per_step,
        )
        self.stop_linear = nn.Linear(hparams.d_model, hparams.n_frames_per_step,)
        self.postnet = Postnet(hparams)

    def parse_output(self, outputs, in_flag=None, out_flag=None):
        if out_flag is not None:
            mel_num = outputs[0].size(1)
            mask = get_output_mask(out_flag)

            # (mel_num, N, L) ---> (N, mel_num, L)
            mask = mask.expand(mel_num, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            # for supporting r frames per step
            N, C, L_r = outputs[0].size()
            L = mask.size(2)
            mask_r = mask.new_ones(N, C, L_r)
            mask_r[:, :, :L] = mask

            outputs[0] = outputs[0].masked_fill(mask_r, 0.0)
            outputs[1] = outputs[1].masked_fill(mask_r, 0.0)
            outputs[2] = outputs[2].masked_fill(mask_r[:, 0, :], 1e3)
        return outputs, in_flag, out_flag

    def forward(self, inputs):

        # parse input
        src_seq, src_flag, trg_seq, trg_flag = inputs
        src_qmask, src_kmask = get_attn_mask(src_flag)
        trg_qmask, trg_kmask = get_attn_mask(trg_flag)
        trg_cmask = get_causal_mask(trg_seq)
        trg_kmask = trg_kmask | trg_cmask

        # encoder
        src_input = self.encoder_prenet(src_seq)
        enc_output, enc_attn_list = self.encoder(
            src_input, src_qmask, src_kmask, return_attns=True
        )

        # decoder
        trg_input = self.decoder_prenet(trg_seq)
        dec_output, dec_attn_list, dec_enc_attn_list = self.decoder(
            trg_input, trg_qmask, trg_kmask, enc_output, src_kmask, return_attns=True,
        )
        mel_output = self.mel_linear(dec_output)
        stop_output = self.stop_linear(dec_output)

        # reshape to original format
        mel_output = mel_output.transpose(1, 2)
        stop_output = stop_output.squeeze(-1)

        # postnet and residual connection
        mel_output_postnet = self.postnet(mel_output)
        mel_output_postnet = mel_output_postnet + mel_output

        # parse output
        outputs = self.parse_output(
            [
                mel_output,
                mel_output_postnet,
                stop_output,
                enc_attn_list,
                dec_attn_list,
                dec_enc_attn_list,
            ],
            src_flag,
            trg_flag,
        )
        return outputs

    def inference(self, inputs):
        src_seq = inputs[0]
        src_qmask, src_kmask = None, None
        trg_qmask = None

        # encoder
        # enc_output (b, enc_l, d_model), enc_attn_list [(b, h, enc_l, enc_l)]
        src_input = self.encoder_prenet(src_seq)
        enc_output, enc_attn_list = self.encoder(
            src_input, src_qmask, src_kmask, return_attns=True
        )

        # create the go frame (b, d_mel, 1)
        trg_go = enc_output.new_full((enc_output.size(0), self.d_mel), 1.0)
        trg_go = trg_go.unsqueeze(-1)

        while True:
            # trg_go (b, d_mel, dec_l), trg_input (b, dec_l, d_model)
            trg_input = self.decoder_prenet(trg_go)
            trg_kmask = get_causal_mask(trg_go)

            # dec_attn_list [(b, h, dec_l, dec_l)]
            # dec_enc_attn_list [(b, h, dec_l, enc_l)]
            dec_output, dec_attn_list, dec_enc_attn_list = self.decoder(
                trg_input,
                trg_qmask,
                trg_kmask,
                enc_output,
                src_kmask,
                return_attns=True,
            )
            mel_output = self.mel_linear(dec_output)
            stop_output = self.stop_linear(dec_output)

            # reshape to original format
            # mel_output (b, d_mel, dec_l), stop_output (b, dec_l)
            mel_output = mel_output.transpose(1, 2)
            stop_output = stop_output.squeeze(-1)

            # create the new input
            trg_go = torch.cat((trg_go, mel_output[:, :, -1:]), dim=-1)

            eos = torch.sigmoid(stop_output).detach().cpu().numpy()
            if True in (eos >= self.stop_threshold):
                break
            elif mel_output.size(-1) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

        # Remove the added go frame and delete the potential noise frame
        mel_output = trg_go[:, :, 1:]
        mel_output = mel_output[:, :, : -self.infer_trim]
        stop_output = stop_output[:, : -self.infer_trim]
        for layer_id in range(len(dec_enc_attn_list)):
            dec_attn = dec_attn_list[layer_id]
            dec_attn = dec_attn[:, :, : -self.infer_trim, : -self.infer_trim]
            dec_attn_list[layer_id] = dec_attn
            dec_enc_attn = dec_enc_attn_list[layer_id]
            dec_enc_attn = dec_enc_attn[:, :, : -self.infer_trim, :]
            dec_enc_attn_list[layer_id] = dec_enc_attn
        
        mel_output_postnet = self.postnet(mel_output)
        mel_output_postnet = mel_output + mel_output_postnet
        outputs = self.parse_output(
            [
                mel_output,
                mel_output_postnet,
                stop_output,
                enc_attn_list,
                dec_attn_list,
                dec_enc_attn_list,
            ]
        )

        assert mel_output.size(2) == dec_enc_attn_list[0].size(2)
        return outputs[0]
