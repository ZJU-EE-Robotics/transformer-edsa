import torch

import numpy as np
import librosa
import scipy

# from scipy.io.wavfile import read

import hparams as hparams


def preemphasis(x):
    return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x):
    return scipy.signal.lfilter([1], [1, -hparams.preemphasis], x)


def load_wav_to_torch(full_path, sr):
    data, sampling_rate = librosa.load(full_path, sr=sr)
    data, _ = librosa.effects.trim(data, top_db=55)
    if hparams.tacotron1_norm is True:
        data = preemphasis(data)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x, gpu=0):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(gpu, non_blocking=True)
    return x


def shift_mel(mel):
    go_frame = mel.new_full((mel.size(0), mel.size(1)), 1.0)
    mel_shift = torch.cat((go_frame.unsqueeze(-1), mel), dim=-1)
    mel_shift = mel_shift[:, :, :-1]
    return mel_shift


def parse_batch(batch, gpu=0):
    (txt_padded, txt_length, mel_padded, gate_padded, mel_length,) = batch
    txt_padded = to_gpu(txt_padded, gpu).long()
    txt_length = to_gpu(txt_length, gpu).long()
    mel_padded = to_gpu(mel_padded, gpu).float()
    gate_padded = to_gpu(gate_padded, gpu).float()
    mel_length = to_gpu(mel_length, gpu).long()

    ref_padded = shift_mel(mel_padded)

    txt_flag = txt_padded.new_zeros(txt_padded.size(0), txt_padded.size(1))
    mel_flag = txt_padded.new_zeros(mel_padded.size(0), mel_padded.size(2))

    for i in range(txt_flag.size(0)):
        txt_flag[i, : txt_length[i]] = 1

    # the added zero frame for EOS is counted
    for i in range(mel_flag.size(0)):
        mel_flag[i, : mel_length[i] + 1] = 1

    return (
        (txt_padded, txt_flag, ref_padded, mel_flag),
        (mel_padded, gate_padded),
    )
