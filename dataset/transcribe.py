from abc import ABC, abstractmethod
import argparse
import os

import librosa
import wave
import numpy as np
import deepspeech
import torch
import torchaudio  # noqa
import omegaconf  # noqa


class TranscriptionModel(ABC):
    @abstractmethod
    def load_audio(self, path):
        pass

    @abstractmethod
    def transcribe(self, path):
        pass


class DeepSpeech(TranscriptionModel):
    """
    Credit: https://github.com/mozilla/DeepSpeech
    """

    def __init__(self, model_path):
        self.model = deepspeech.Model(model_path)

    def load_audio(self, path):
        try:
            audio = wave.open(path, "r")
        except Exception:
            raise Exception(f"Cannot load audio file {path}")

        frames = audio.getnframes()
        buffer = audio.readframes(frames)
        return np.frombuffer(buffer, dtype=np.int16)

    def transcribe(self, path):
        assert os.path.isfile(path), f"{path} not found. Cannot transcribe"

        data = self.load_audio(path)
        output = self.model.stt(data)
        return output


class Silero(TranscriptionModel):
    """
    Credit: https://github.com/snakers4/silero-models
    """

    def __init__(self, language="en"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.decoder, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models", model="silero_stt", language=language, device=self.device
        )

    def load_audio(self, path):
        try:
            wav, _ = librosa.load(path, sr=16000)
        except Exception:
            raise Exception(f"Cannot load audio file {path}")

        assert len(wav) > 0, f"{path} wav file is empty"
        return torch.tensor([wav])

    def transcribe(self, path):
        assert os.path.isfile(path), f"{path} not found. Cannot transcribe"
        data = self.load_audio(path)
        data = data.to(self.device)
        output = self.model(data)

        for example in output:
            return self.decoder(example.cpu())


def create_transcription_model(model_path=None):
    if model_path:
        return DeepSpeech(model_path)
    # If no model path, default to English Sliero
    else:
        return Silero()


if __name__ == "__main__":
    """Transcribe a clip"""
    parser = argparse.ArgumentParser(description="Transcribe a clip")
    parser.add_argument("-i", "--input_path", help="Path to audio file", type=str, required=True)
    args = parser.parse_args()

    text = transcribe(args.input_path)
    print("Text: ", text)
