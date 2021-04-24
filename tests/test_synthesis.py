import os
import inflect

from synthesis.synthesize import load_model, synthesize
from synthesis.waveglow import load_waveglow_model
from dataset.transcribe import transcribe


def text_similarity(a, b):
    return 1 - (len(set(a.split(" ")) - set(b.split(" "))) / len(a.split(" ")))


def test_synthesize():
    model_path = os.path.join("files", "tacotron2_statedict.pt")
    waveglow_path = os.path.join("files", "waveglow_256channels_universal_v5.pt")
    graph_path = "graph.png"
    audio_path = "synthesized_audio.wav"

    model = load_model(model_path)
    assert model

    waveglow = load_waveglow_model(waveglow_path)
    assert waveglow

    text = "hello everybody my name is david attenborough"
    inflect_engine = inflect.engine()
    synthesize(model, waveglow, text, inflect_engine, graph=graph_path, audio=audio_path)

    assert text_similarity(text, transcribe(audio_path)) > 0.5
    assert os.path.isfile(graph_path)
    assert os.path.isfile(audio_path)

    os.remove(graph_path)
    os.remove(audio_path)
