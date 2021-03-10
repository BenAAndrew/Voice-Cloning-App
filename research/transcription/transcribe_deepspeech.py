import torch
import os
import wave
import numpy as np
import sys
import shlex
import subprocess
from deepspeech import Model
from tqdm import tqdm

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

model = Model("deepspeech-0.9.3-models.pbmm")
model.enableExternalScorer("deepspeech-0.9.3-models.scorer")
desired_sample_rate = model.sampleRate()
PATH = os.path.join("LJSpeech-1.1", "wavs")
TOTAL_SAMPLES = 100


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = "sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - ".format(
        quote(audio_path), desired_sample_rate
    )
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))
    except OSError as e:
        raise OSError(
            e.errno, "SoX not found, use {}hz files or install it: {}".format(desired_sample_rate, e.strerror)
        )

    return desired_sample_rate, np.frombuffer(output, np.int16)


def load_audio(path):
    fin = wave.open(path, "rb")
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print(
            "Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.".format(
                fs_orig, desired_sample_rate
            ),
            file=sys.stderr,
        )
        fs_new, audio = convert_samplerate(path, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1 / fs_orig)
    fin.close()
    return audio


def transcribe(path):
    audio = load_audio(path)
    return model.stt(audio)


files = os.listdir(PATH)[:TOTAL_SAMPLES]
sentences = {}

for f in tqdm(files):
    sentences[f] = transcribe(os.path.join(PATH, f))

with open("transcribe_deepspeech.csv", "w") as f:
    for key, value in sentences.items():
        f.write(f"{key}|{value}\n")
