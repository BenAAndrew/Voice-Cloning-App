import os
import inflect
import numpy as np
import librosa
import pytest

from dataset.forced_alignment.search import similarity
from dataset.transcribe import create_transcription_model
from synthesis.synthesize import load_model, synthesize
from synthesis.vocoders import Waveglow, Hifigan
from synthesis.vocoders.vocoder import Vocoder


MIN_SYNTHESIS_SCORE = 0.3


class FakeVocoder(Vocoder):
    def generate_audio(self, mel_output):
        # 1 second of silence
        return np.zeros(22050).astype("int16")


@pytest.mark.slow
def test_synthesis():
    model_path = os.path.join("test_samples", "model.pt")
    graph_path = "graph.png"
    audio_path = "synthesized_audio.wav"

    model = load_model(model_path)
    vocoder = FakeVocoder()
    inflect_engine = inflect.engine()

    # Single line
    text = "hello everybody my name is david attenborough"
    synthesize(
        model=model,
        text=text,
        inflect_engine=inflect_engine,
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
    text = (
        "The monkeys live in the jungle with their families.\nHowever, I prefer to live on the beach and enjoy the sun."
    )
    synthesize(
        model=model,
        text=text,
        inflect_engine=inflect_engine,
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


@pytest.mark.slow
def test_waveglow_synthesis():
    model_path = os.path.join("test_samples", "model.pt")
    waveglow_path = os.path.join("test_samples", "waveglow_256channels_universal_v5.pt")
    graph_path = "graph.png"
    audio_path = "synthesized_audio.wav"
    transcription_model = create_transcription_model()

    model = load_model(model_path)
    waveglow = Waveglow(waveglow_path)
    text = "hello everybody my name is david attenborough"
    inflect_engine = inflect.engine()
    synthesize(
        model=model,
        text=text,
        inflect_engine=inflect_engine,
        graph_path=graph_path,
        audio_path=audio_path,
        vocoder=waveglow,
    )

    assert os.path.isfile(graph_path)
    assert os.path.isfile(audio_path)
    assert similarity(text, transcription_model.transcribe(audio_path)) > MIN_SYNTHESIS_SCORE

    os.remove(graph_path)
    os.remove(audio_path)


@pytest.mark.slow
def test_hifigan_synthesis():
    model_path = os.path.join("test_samples", "model.pt")
    hifigan_model_path = os.path.join("test_samples", "hifigan.pt")
    hifigan_config_path = os.path.join("test_samples", "config.json")
    graph_path = "graph.png"
    audio_path = "synthesized_audio.wav"
    transcription_model = create_transcription_model()

    model = load_model(model_path)
    hifigan = Hifigan(hifigan_model_path, hifigan_config_path)
    text = "hello everybody my name is david attenborough"
    inflect_engine = inflect.engine()
    synthesize(
        model=model,
        text=text,
        inflect_engine=inflect_engine,
        graph_path=graph_path,
        audio_path=audio_path,
        vocoder=hifigan,
    )

    assert os.path.isfile(graph_path)
    assert os.path.isfile(audio_path)
    assert similarity(text, transcription_model.transcribe(audio_path)) > MIN_SYNTHESIS_SCORE

    os.remove(graph_path)
    os.remove(audio_path)
