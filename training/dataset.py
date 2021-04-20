import os
import numpy as np
import random
import torch
from tacotron2_model import TacotronSTFT
from scipy.io.wavfile import read
import inflect
from unidecode import unidecode

from training.clean_text import clean_text


def load_wav_to_torch(full_path):
    """
    Credit: https://github.com/NVIDIA/tacotron2

    Loads wav file to FloatTensor.

    Parameters
    ----------
    full_path : str
        Path to audio file
    
    Returns
    -------
    FloatTensor
        Tensor containing wav data
    int
        Sample rate
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class VoiceDataset(torch.utils.data.Dataset):
    """
    Credit: https://github.com/NVIDIA/tacotron2

    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """
    def __init__(self, filepaths_and_text, dataset_path, symbols, seed):
        self.filepaths_and_text = filepaths_and_text
        self.dataset_path = dataset_path
        self.inflect_engine = inflect.engine()
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.load_mel_from_disk = False
        self.stft = TacotronSTFT()
        random.shuffle(self.filepaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        filepath = os.path.join(self.dataset_path, filename)

        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filepath)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filepath))
            assert melspec.size(0) == self.stft.n_mel_channels, "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )

        return melspec

    def get_text(self, text):
        text = clean_text(text, self.inflect_engine)
        sequence = [self.symbol_to_id[s] for s in text if s != "_"]
        text_norm = torch.IntTensor(sequence)
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.filepaths_and_text[index])

    def __len__(self):
        return len(self.filepaths_and_text)
