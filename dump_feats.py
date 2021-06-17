import torch
from torch.utils.data import Dataset, DataLoader

import os
import librosa
import numpy as np
from tqdm import tqdm

from stft import TacotronSTFT
from utils_public import preemphasis
import hparams as hparams


class DumpFeats(Dataset):
    """
    Preparing the wav files to mel spectrograms and saving them on disk for
    speeding up training. The generated mel spectrograms are saved in the same
    path with .wav files.
    """

    def __init__(self, wav_path, data_path):
        self.wav_list = []
        self.wav_path = wav_path
        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )

        with open(data_path, "r") as f:
            for line in f:
                line = line.split("|")
                self.wav_list.append(line[0].strip())

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        return self._get_mel(self.wav_list[index])

    def _get_mel(self, file):
        wav = os.path.join(self.wav_path, file + ".wav")
        mel = os.path.join(self.wav_path, file + ".mel.npy")

        # Loading sound file
        audio, sr = librosa.load(wav, sr=hparams.sampling_rate)

        # Trimming
        audio, _ = librosa.effects.trim(audio, top_db=55)

        # which normalization to use, Tacotron1 or Tacotron2
        if hparams.tacotron1_norm is True:
            audio = preemphasis(audio)

        # stft, using the same stft with utils_data.py
        audio_norm = torch.FloatTensor(audio.astype(np.float32)).unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.squeeze(0)
        np.save(mel, melspec.numpy())
        return melspec

    def compute_statistics(self, train_set):
        feats_sum = np.zeros((80, 1), dtype=np.float64)
        frames_sum = 0

        # mean
        with open(train_set, "r") as f:
            for line in tqdm(f, desc="compute the mean: "):
                line = line.split("|")
                feat = np.load(os.path.join(self.wav_path, line[0] + ".mel.npy"))
                feat = feat.astype(np.float64)
                feats_sum = feats_sum + np.sum(feat, axis=-1, keepdims=True)
                frames_sum = frames_sum + feat.shape[-1]
        feats_mean = feats_sum / frames_sum

        # std
        feats_err_sum = np.zeros((80, 1), dtype=np.float64)
        with open(train_set, "r") as f:
            for line in tqdm(f, desc="compute the std: "):
                line = line.split("|")
                feat = np.load(os.path.join(self.wav_path, line[0] + ".mel.npy"))
                feat = feat.astype(np.float64)
                feats_err = np.power((feat - feats_mean), 2)
                feats_err = np.sum(feats_err, axis=-1, keepdims=True)
                feats_err_sum = feats_err_sum + feats_err
        feats_var = feats_err_sum / frames_sum
        feats_std = np.power(feats_var, 0.5)

        feats_mean = feats_mean.astype(np.float32)
        feats_std = feats_std.astype(np.float32)
        feats_cmvn = np.concatenate((feats_mean, feats_std), axis=-1)

        # save statistics
        np.save(os.path.join(self.wav_path, "cmvn.npy"), feats_cmvn)

    def normalize_feats(self, data_path):
        cmvn = np.load(os.path.join(self.wav_path, "cmvn.npy"))
        mean = cmvn[:, 0:1]
        std = cmvn[:, 1:]
        with open(data_path, "r") as f:
            for line in tqdm(f, desc="normalize the feats: "):
                line = line.split("|")
                feat = np.load(os.path.join(self.wav_path, line[0] + ".mel.npy"))
                norm_feat = (feat - mean) / std
                np.save(os.path.join(self.wav_path, line[0] + ".norm.npy"), norm_feat)


if __name__ == "__main__":

    # path should be redefined for your own
    wav_path = "/home/server/disk1/DATA/LJS/LJSpeech-1.1/wavs"
    data_path = "filelists/data.csv"
    train_set = "filelists/train_set.csv"

    dataset = DumpFeats(wav_path, data_path)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
    for _ in tqdm(dataloader, desc="Dumping the raw feats: "):
        pass

    dataset.compute_statistics(train_set)
    dataset.normalize_feats(data_path)
    print("Dumping the feats is finished !")
