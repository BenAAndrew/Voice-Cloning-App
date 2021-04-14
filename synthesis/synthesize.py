import argparse
import os
import inflect
import matplotlib.pyplot as plt
import IPython.display as ipd
from tacotron2_model import Tacotron2
import torch
import numpy as np
import glow
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

from training.clean_text import clean_text


SYMBOLS = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}


def load_model(model_path):
    if torch.cuda.is_available():
        model = Tacotron2().cuda()
        model.load_state_dict(torch.load(model_path)["state_dict"])
        _ = model.cuda().eval().half()
    else:
        model = Tacotron2()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["state_dict"])
    return model


def load_waveglow(waveglow_path):
    waveglow = torch.load(waveglow_path)["model"]
    if torch.cuda.is_available():
        waveglow.cuda().eval().half()

    for k in waveglow.convinv:
        k.float()
    return waveglow


def generate_graph(alignments, filepath):
    data = alignments.float().data.cpu().numpy()[0].T
    plt.imshow(data, aspect="auto", origin="lower", interpolation="none")
    plt.savefig(filepath)


def generate_audio(mel, waveglow, filepath, sample_rate=22050):
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=0.666)

    audio = audio[0].data.cpu().numpy()
    audio = ipd.Audio(audio, rate=sample_rate)
    with open(filepath, "wb") as f:
        f.write(audio.data)


def text_to_sequence(text):
    sequence = np.array([[SYMBOL_TO_ID[s] for s in text if s in SYMBOL_TO_ID]])
    if torch.cuda.is_available():
        return torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    else:
        return torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()


def synthesize(model, waveglow_model, text, inflect_engine, graph=None, audio=None):
    text = clean_text(text, inflect_engine)
    sequence = text_to_sequence(text)
    _, mel_outputs_postnet, _, alignments = model.inference(sequence)

    if graph:
        generate_graph(alignments, graph)

    if audio:
        generate_audio(mel_outputs_postnet, waveglow_model, audio)


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
    waveglow_model = load_waveglow(args.waveglow_path)
    inflect_engine = inflect.engine()

    synthesize(
        model,
        waveglow_model,
        args.text,
        inflect_engine,
        args.graph_output_path,
        args.audio_output_path,
    )
