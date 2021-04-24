import argparse
import os
import inflect
import matplotlib.pyplot as plt
from tacotron2_model import Tacotron2
import torch
import numpy as np
import glow
import matplotlib
from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
matplotlib.use("Agg")

from training.clean_text import clean_text
from synthesis.waveglow import load_waveglow_model, generate_audio_waveglow
from synthesis.hifigan import generate_audio_hifigan


SYMBOLS = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


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


def text_to_sequence(text):
    """
    Generates synthesised audio file.

    Parameters
    ----------
    mel : list
        Synthesised mel data
    waveglow : Torch
        Waveglow model
    filepath : str
        Path to save generated audio to
    sample_rate : int (optional)
        Sample rate of audio (default is 22050)
    """
    sequence = np.array([[SYMBOL_TO_ID[s] for s in text if s in SYMBOL_TO_ID]])
    if torch.cuda.is_available():
        return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    else:
        return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def synthesize(model, text, inflect_engine, graph=None, audio=None, vocoder=None, vocoder_type=None):
    """
    Synthesise text for a given model.
    Produces graph and/or audio file when given.

    Parameters
    ----------
    model : Tacotron2
        Tacotron2 model
    waveglow_model : Torch
        Waveglow model
    text : str
        Text to synthesize
    inflect_engine : Inflect
        Inflect.engine() object
    graph : str (optional)
        Path to save alignment graph to
    audio : str (optional)
        Path to save audio file to
    """
    text = clean_text(text, inflect_engine)
    sequence = text_to_sequence(text)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, help="tacotron2 model path", required=True)
    parser.add_argument("-w", "--waveglow_model_path", type=str, help="waveglow model path", required=True)
    parser.add_argument("-t", "--text", type=str, help="text to synthesize", required=True)
    parser.add_argument("-g", "--graph_output_path", type=str, help="path to save alignment graph to", required=False)
    parser.add_argument("-a", "--audio_output_path", type=str, help="path to save output audio to", required=False)
    args = parser.parse_args()

    assert os.path.isfile(args.model_path), "Model not found"
    assert os.path.isfile(args.waveglow_model_path), "Waveglow model not found"

    model = load_model(args.model_path)
    waveglow_model = load_waveglow_model(args.waveglow_path)
    inflect_engine = inflect.engine()

    synthesize(
        model, args.text, inflect_engine, args.graph_output_path, args.audio_output_path, waveglow_model, "waveglow"
    )
