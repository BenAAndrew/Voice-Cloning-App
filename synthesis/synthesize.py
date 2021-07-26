import argparse
import os
import inflect
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib
from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
matplotlib.use("Agg")

import glow  # noqa
from training.tacotron2_model import Tacotron2
from training.clean_text import clean_text
from training.train import DEFAULT_ALPHABET
from synthesis.waveglow import load_waveglow_model, generate_audio_waveglow
from synthesis.hifigan import load_hifigan_model, generate_audio_hifigan


def load_model(model_path):
    """
    Loads the Tacotron2 model.
    Uses GPU if available, otherwise uses CPU.

    Parameters
    ----------
    model_path : str
        Path to tacotron2 model

    Returns
    -------
    Tacotron2
        Loaded tacotron2 model
    """
    if torch.cuda.is_available():
        model = Tacotron2().cuda()
        model.load_state_dict(torch.load(model_path)["state_dict"])
        _ = model.cuda().eval().half()
    else:
        model = Tacotron2()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["state_dict"])
    return model


def generate_graph(alignments, filepath):
    """
    Generates synthesis alignment graph image.

    Parameters
    ----------
    alignments : list
        Numpy alignment data
    filepath : str
        Path to save image to
    """
    data = alignments.float().data.cpu().numpy()[0].T
    plt.imshow(data, aspect="auto", origin="lower", interpolation="none")
    plt.savefig(filepath)


def text_to_sequence(text, symbols):
    """
    Generates text sequence for audio file

    Parameters
    ----------
    text : str
        Text to synthesize
    symbols : list
        List of valid symbols
    """
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sequence = np.array([[symbol_to_id[s] for s in text if s in symbol_to_id]])
    if torch.cuda.is_available():
        return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    else:
        return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def synthesize(model, text, inflect_engine, symbols=DEFAULT_ALPHABET, graph=None, audio=None, vocoder=None, vocoder_type=None):
    """
    Synthesise text for a given model.
    Produces graph and/or audio file when given.

    Parameters
    ----------
    model : Tacotron2
        Tacotron2 model
    text : str
        Text to synthesize
    inflect_engine : Inflect
        Inflect.engine() object
    symbols : list
        List of symbols (default is English)
    graph : str (optional)
        Path to save alignment graph to
    audio : str (optional)
        Path to save audio file to
    vocoder : Object (optional)
        Vocoder model (required if generating audio)
    vocoder_type : str (optional)
        Vocoder type (required if generating audio)
    """
    text = clean_text(text, inflect_engine)
    sequence = text_to_sequence(text, symbols)
    _, mel_outputs_postnet, _, alignments = model.inference(sequence)

    if graph:
        generate_graph(alignments, graph)

    if audio:
        assert vocoder, "Missing vocoder"
        if vocoder_type == "hifigan":
            generate_audio_hifigan(vocoder, mel_outputs_postnet, audio)
        elif vocoder_type == "waveglow":
            generate_audio_waveglow(vocoder, mel_outputs_postnet, audio)
        else:
            raise Exception(f"Unsupported vocoder type {vocoder_type}")


if __name__ == "__main__":
    """Synthesize audio using model and vocoder"""
    parser = argparse.ArgumentParser(description="Synthesize audio using model and vocoder")
    parser.add_argument("-m", "--model_path", type=str, help="tacotron2 model path", required=True)
    parser.add_argument("-vt", "--vocoder_type", type=str, help="vocoder type(waveglow or hifigan)", required=True)
    parser.add_argument("-vm", "--vocoder_model_path", type=str, help="vocoder model path", required=True)
    parser.add_argument("-hc", "--hifigan_config_path", type=str, help="hifigan_config path", required=False)
    parser.add_argument("-t", "--text", type=str, help="text to synthesize", required=True)
    parser.add_argument("-g", "--graph_output_path", type=str, help="path to save alignment graph to", required=False)
    parser.add_argument("-a", "--audio_output_path", type=str, help="path to save output audio to", required=False)
    args = parser.parse_args()

    assert os.path.isfile(args.model_path), "Model not found"
    assert os.path.isfile(args.vocoder_model_path), "vocoder model not found"

    model = load_model(args.model_path)
    vocoder_type = args.vocoder_type
    vocoder_model = None
    if vocoder_type == "hifigan":
        assert os.path.isfile(args.hifigan_config_path), "hifigan config not found"
        vocoder_model = load_hifigan_model(args.vocoder_model_path, args.hifigan_config_path)
    elif vocoder_type == "waveglow":
        vocoder_model = load_waveglow_model(args.vocoder_model_path)

    inflect_engine = inflect.engine()

    synthesize(
        model, args.text, inflect_engine, args.graph_output_path, args.audio_output_path, vocoder_model, vocoder_type
    )
