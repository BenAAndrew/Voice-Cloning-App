import argparse
import os

from tqdm import tqdm
import numpy as np
import torch
from scipy.io.wavfile import read
from tacotron2_model.stft import TacotronSTFT

max_wav_value = 32768.0
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def wav_to_mel(stft, path, output_path):
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, stft.sampling_rate))
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    np.save(output_path, melspec, allow_pickle=False)


if __name__ == "__main__":
    """ Script to generate MELs from wavs """
    parser = argparse.ArgumentParser(description="Convert WAVs to MEL spectograms")
    parser.add_argument("-w", "--wavs", help="Text file path", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    stft = TacotronSTFT(filter_length, hop_length, win_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax)

    for f in tqdm(os.listdir(args.wavs)):
        wav_path = os.path.join(args.wavs, f)
        output_path = os.path.join(args.output, f.replace(".wav", ".npy"))
        wav_to_mel(stft, wav_path, output_path)
