import argparse
import torch
import os
import wave
import numpy as np


def load_audio(path):
    """
    Loads the audio from a given path into an array for transcription.

    Parameters
    ----------
    path : str
        Path to audio file

    Raises
    -------
    Exception
        If the audio file could not be loaded
    AssertionError
        If the audio file was empty
    """
    try:
        audio = wave.open(path, 'r')
    except Exception:
        raise Exception(f"Cannot load audio file {path}")

    frames = audio.getnframes()
    buffer = audio.readframes(frames)
    return np.frombuffer(buffer, dtype=np.int16)


def transcribe(path, model):
    """
    Credit: https://github.com/mozilla/DeepSpeech

    Transcribes a given audio file.

    Parameters
    ----------
    path : str
        Path to audio file
    model : Deepspeech model
        Deepspeech model

    Raises
    -------
    Exception
        If the audio file could not be loaded
    AssertionError
        If the audio file was not found or was empty

    Returns
    -------
    str
        Text transcription of audio file
    """
    assert os.path.isfile(path), f"{path} not found. Cannot transcribe"

    data = load_audio(path)
    output = model.stt(data)
    return output


if __name__ == "__main__":
    """Transcribe a clip"""
    parser = argparse.ArgumentParser(description="Transcribe a clip")
    parser.add_argument("-i", "--input_path", help="Path to audio file", type=str, required=True)
    args = parser.parse_args()

    text = transcribe(args.input_path)
    print("Text: ", text)
