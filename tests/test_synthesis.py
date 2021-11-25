import os
import numpy as np
import librosa
import torch

from dataset.utils import similarity
from dataset.transcribe import Silero
from synthesis.synthesize import load_model, synthesize
from synthesis.vocoders import Hifigan
from synthesis.vocoders.vocoder import Vocoder
from training.tacotron2_model.model import Tacotron2


MIN_SYNTHESIS_SCORE = 0.3


class FakeVocoder(Vocoder):
    # 1 second of silence
    audio = np.zeros(22050).astype("int16")

    def generate_audio(self, mel_output):
        return self.audio


class FakeModel:
    mel_output = torch.zeros((1, 80, 10), dtype=torch.float16)
    alignment = torch.zeros((1, 10, 5), dtype=torch.float16)

    def inference(self, sequence, max_decoder_steps):
        return None, self.mel_output, None, self.alignment


def test_synthesize():
    graph_path = "graph.png"
    audio_path = "synthesized_audio.wav"

    model = FakeModel()
    vocoder = FakeVocoder()

    # Single line
    text = "hello everybody my name is david attenborough"
    synthesize(
        model=model,
        text=text,
        graph_path=graph_path,
        audio_path=audio_path,
        vocoder=vocoder,
        sample_rate=22050,
    )

    assert os.path.isfile(graph_path)
    assert os.path.isfile(audio_path)
    assert librosa.get_duration(filename=audio_path) == 1

    os.remove(graph_path)
    os.remove(audio_path)

    # Multi line
    text = [
        "the monkeys live in the jungle with their families.",
        "however, i prefer to live on the beach and enjoy the sun.",
    ]
    synthesize(
        model=model,
        text=text,
        graph_path=graph_path,
        audio_path=audio_path,
        vocoder=vocoder,
        silence_padding=0.5,
        sample_rate=22050,
    )

    assert os.path.isfile(graph_path)
    assert os.path.isfile(audio_path)
    assert librosa.get_duration(filename=audio_path) == 2.5

    os.remove(graph_path)
    os.remove(audio_path)

    # Split text
    text = (
        "the monkeys live in the jungle with their families. however, i prefer to live on the beach and enjoy the sun."
    )
    synthesize(
        model=model,
        text=text,
        graph_path=None,
        audio_path=audio_path,
        vocoder=vocoder,
        silence_padding=0.5,
        sample_rate=22050,
        split_text=True,
    )

    assert os.path.isfile(audio_path)
    assert librosa.get_duration(filename=audio_path) == 2.5

    os.remove(audio_path)


class FakeModelForSynthesis:
    mel_output = torch.load(os.path.join("test_samples", "mel.pt"))

    def inference(self, sequence, max_decoder_steps):
        return None, self.mel_output, None, None


def test_hifigan_synthesis():
    hifigan_model_path = os.path.join("test_samples", "hifigan.pt")
    hifigan_config_path = os.path.join("test_samples", "config.json")
    audio_path = "synthesized_audio.wav"
    transcription_model = Silero()

    hifigan = Hifigan(hifigan_model_path, hifigan_config_path)
    text = "the monkeys live"
    synthesize(
        model=FakeModelForSynthesis(),
        text=text,
        graph_path=None,
        audio_path=audio_path,
        vocoder=hifigan,
    )

    assert os.path.isfile(audio_path)
    assert similarity(text, transcription_model.transcribe(audio_path)) > MIN_SYNTHESIS_SCORE

    os.remove(audio_path)


def test_load_model():
    model_path = os.path.join("test_samples", "model.pt")
    model = load_model(model_path)
    assert isinstance(model, Tacotron2)
