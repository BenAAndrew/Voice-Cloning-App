from abc import ABC, abstractmethod
import argparse
import os
import sys
import librosa
import wave
import numpy as np
import deepspeech
import torch
import torchaudio  # noqa
import soundfile  # noqa
import omegaconf  # noqa

from dataset.silero_utils import init_jit_model


SILERO_LANGUAGES = {"English": "en", "German": "de", "Spanish": "es"}


class TranscriptionModel(ABC):
    @abstractmethod
    def load_audio(self, path):
        """
        Loads the audio file into a format that can be handled
        by the transcribe function

        Parameters
        ----------
        path : str
            Path to audio file

        Returns
        -------
        list or np.array
            Loaded audio file
        """
        pass

    @abstractmethod
    def transcribe(self, path):
        """
        Transcribes a given audio file.

        Parameters
        ----------
        path : str
            Path to audio file

        Returns
        -------
        str
            Text transcription
        """
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

    def __init__(self, language="English"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = os.path.join(getattr(sys, "_MEIPASS", ""), "en_v5.jit")
        if os.path.isfile(model):
            self.model, self.decoder = init_jit_model(model, self.device)
        else:
            self.model, self.decoder, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_stt",
                language=SILERO_LANGUAGES[language],
                device=self.device,
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


if __name__ == "__main__":
    """Transcribe a clip"""
    parser = argparse.ArgumentParser(description="Transcribe a clip")
    parser.add_argument("-i", "--input_path", help="Path to audio file", type=str, required=True)
    args = parser.parse_args()

    model = Silero()
    text = model.transcribe(args.input_path)
    print("Text: ", text)
