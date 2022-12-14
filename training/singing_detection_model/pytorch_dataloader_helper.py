import torch
import torchaudio
import pandas as pd
import torchvision
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset


class CustomAudioDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, training, transform=None):
        if training:
            self.data = pd.read_csv("train.csv")
        else:
            self.data = pd.read_csv("valid.csv")

        self.transform = transform
        self.new_rate = 16000
        self.max_num_frames = 30 * self.new_rate
        self.target_size = (64, 256)

        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.loc[idx, "wavfile"]

        # read file and resample
        signal, sample_rate = torchaudio.load(filename)
        signal = torch.mean(signal, axis=0)[None, :]

        if sample_rate != self.new_rate:
            resampler = T.Resample(sample_rate, self.new_rate, dtype=signal.dtype)
            resampled_waveform = resampler(signal)
        else:
            resampled_waveform = signal

        # padding
        if resampled_waveform.shape[1] > self.max_num_frames:
            padded = resampled_waveform[:, 0 : self.max_num_frames]
        elif resampled_waveform.shape[1] < self.max_num_frames:
            num_pad = self.max_num_frames - resampled_waveform.shape[1]
            padded = F.pad(resampled_waveform, (0, num_pad), mode="constant")

        melspec = self.mel_spectrogram(padded)
        resized_melspec = torchvision.transforms.Resize(size=self.target_size)(melspec)

        if self.transform:
            sample = self.transform(sample)

        return {
            "image": resized_melspec,
            "is_speech": self.data.loc[idx, "label"],
        }
