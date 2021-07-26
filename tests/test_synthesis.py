import os
import inflect

from tests import download_if_not_exists
from dataset.forced_alignment.search import similarity
from dataset.transcribe import create_transcription_model
from synthesis.synthesize import load_model, synthesize
from synthesis.waveglow import load_waveglow_model
from synthesis.hifigan import load_hifigan_model


TACOTRON2_ID = "12f-ibcqe1x3HQ_OvyuNLMRr4ir5DTjRf"
WAVEGLOW_ID = "1iuxOwcHHfuLpwfAKSoZVQmNryzH9n4W6"
HIFIGAN_ID = "1hmPWhmEcdlbevCaFO7isYFwnaFUa0l7u"
HIFIGAN_CONFIG_ID = "1XZNMtaTw7AyBYAvHDEg0Hybole8ffThL"


def test_waveglow_synthesis():
    model_path = os.path.join("test_samples", "tacotron2_statedict.pt")
    download_if_not_exists(model_path, TACOTRON2_ID)
    waveglow_path = os.path.join("test_samples", "waveglow_256channels_universal_v5.pt")
    download_if_not_exists(waveglow_path, WAVEGLOW_ID)
    graph_path = "graph.png"
    audio_path = "synthesized_audio.wav"
    transcription_model = create_transcription_model()

    model = load_model(model_path)
    assert model

    waveglow = load_waveglow_model(waveglow_path)
    assert waveglow

    text = "hello everybody my name is david attenborough"
    inflect_engine = inflect.engine()
    synthesize(model=model, text=text, inflect_engine=inflect_engine, graph=graph_path, audio=audio_path, vocoder=waveglow, vocoder_type="waveglow")

    assert os.path.isfile(graph_path)
    assert os.path.isfile(audio_path)
    assert similarity(text, transcription_model.transcribe(audio_path)) > 0.5

    os.remove(graph_path)
    os.remove(audio_path)


def test_hifigan_synthesis():
    model_path = os.path.join("test_samples", "tacotron2_statedict.pt")
    download_if_not_exists(model_path, TACOTRON2_ID)
    hifigan_model_path = os.path.join("test_samples", "hifigan.pt")
    download_if_not_exists(hifigan_model_path, HIFIGAN_ID)
    hifigan_config_path = os.path.join("test_samples", "config.json")
    download_if_not_exists(hifigan_config_path, HIFIGAN_CONFIG_ID)
    graph_path = "graph.png"
    audio_path = "synthesized_audio.wav"
    transcription_model = create_transcription_model()

    model = load_model(model_path)
    assert model

    waveglow = load_hifigan_model(hifigan_model_path, hifigan_config_path)
    assert waveglow

    text = "hello everybody my name is david attenborough"
    inflect_engine = inflect.engine()
    synthesize(model=model, text=text, inflect_engine=inflect_engine, graph=graph_path, audio=audio_path, vocoder=waveglow, vocoder_type="hifigan")

    assert os.path.isfile(graph_path)
    assert os.path.isfile(audio_path)
    assert similarity(text, transcription_model.transcribe(audio_path)) > 0.5

    os.remove(graph_path)
    os.remove(audio_path)
