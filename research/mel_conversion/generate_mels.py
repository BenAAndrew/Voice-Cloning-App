import argparse
import os

import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
import numpy as np

SILENCE_THRESHOLD = 60
FRAME_LENGTH = 2048
WIN_LENGTH = 1024
HOP_LENGTH = 512
HOP_LENGTH_2 = 256
N_FFT = 1024
NUM_MELS = 80
FMIN = 0
FMAX = 8000


def wav_to_mel(path, output_path, sample_rate):
    wav = librosa.load(path, sr=sample_rate)[0]
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    # Trim silence
    wav = librosa.effects.trim(wav, top_db=SILENCE_THRESHOLD, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    # Convert to MEL
    D = librosa.stft(y=wav, n_fft=N_FFT, hop_length=HOP_LENGTH_2, win_length=WIN_LENGTH)
    S = librosa.feature.melspectrogram(S=np.abs(D), sr=sample_rate, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX)
    # Normalise
    S = np.clip(S, a_min=1.0e-5, a_max=None)
    S = np.log(S)
    # Save
    np.save(output_path, S, allow_pickle=False)


if __name__ == "__main__":
    """ Script to generate MELs from wavs """
    parser = argparse.ArgumentParser(description="Convert WAVs to MEL spectograms")
    parser.add_argument("-w", "--wavs", help="Text file path", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output path", type=str, required=True)
    parser.add_argument("--sample_rate", help="Audio sample rate", type=int, default=22050)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for f in tqdm(os.listdir(args.wavs)):
        wav_path = os.path.join(args.wavs, f)
        output_path = os.path.join(args.output, f.replace(".wav", ".npy"))
        wav_to_mel(wav_path, output_path, args.sample_rate)
