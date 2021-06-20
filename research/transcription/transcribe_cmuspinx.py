import os
import speech_recognition as sr
from tqdm import tqdm

r = sr.Recognizer()
PATH = os.path.join("LJSpeech-1.1", "wavs")
TOTAL_SAMPLES = 100


def transcribe(path):
    with sr.AudioFile(path) as recording:
        audio = r.record(recording)
        return r.recognize_sphinx(audio)


files = os.listdir(PATH)[:TOTAL_SAMPLES]
sentences = {}

for f in tqdm(files):
    sentences[f] = transcribe(os.path.join(PATH, f))

with open("transcribe_cmuspinx.csv", "w") as f:
    for key, value in sentences.items():
        f.write(f"{key}|{value}\n")
