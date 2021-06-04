import os
import requests

ENGLISH_MODEL_URL = "https://github.com/coqui-ai/STT-models/releases/download/english/coqui/v0.9.3/model.pbmm"
ENGLISH_ALPHABET_URL = "https://github.com/coqui-ai/STT-models/releases/download/english/coqui/v0.9.3/alphabet.txt"
TRANSCRIPTION_MODEL = "model.pbmm"
ALPHABET_FILE = "alphabet.txt"


def download_english(paths):
    """Downloads English transcription model"""
    language_path = os.path.join(paths["languages"], "English")

    if not os.path.isdir(language_path):
        print("DOWNLOADING ENGLISH MODEL")
        os.makedirs(language_path)
        r = requests.get(ENGLISH_MODEL_URL)
        with open(os.path.join(language_path, TRANSCRIPTION_MODEL), "wb") as f:
            f.write(r.content)

        r = requests.get(ENGLISH_ALPHABET_URL)
        with open(os.path.join(language_path, ALPHABET_FILE), "wb") as f:
            f.write(r.content)
