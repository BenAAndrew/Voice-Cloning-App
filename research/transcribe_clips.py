import argparse
import os
import re

from tqdm import tqdm

from dataset.transcribe import transcribe

REGEX = re.compile("[^a-zA-Z ]")
PATH = os.path.join("LJSpeech-1.1", "wavs")
TOTAL_SAMPLES = 100


class Transcription:
    def __init__(self, filename, prediction, actual, score):
        self.filename = filename
        self.prediction = prediction
        self.actual = actual
        self.score = score


def compare(expected_text, actual_text):
    expected_text = REGEX.sub("", expected_text.strip()).lower()
    actual_text = REGEX.sub("", actual_text.strip()).lower()
    expected_words = expected_text.split(" ")
    actual_words = actual_text.split(" ")
    difference = set(expected_words) - set(actual_words)
    return 1 - (len(difference) / len(expected_words))


def save_results(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(f"{item.filename}|{item.actual}|{item.prediction}|{item.score}\n")


def read_labels(path):
    with open(path, encoding="utf-8") as f:
        data = {}
        for line in f.readlines():
            row = line.split("|")
            data[row[0]] = row[1]
        return data


def transcribe_clips(folder, labels, output_path):
    files = os.listdir(folder)[:5]
    labels = read_labels(labels)
    data = []
    for filename in tqdm(files):
        prediction = transcribe(os.path.join(folder, filename))
        actual = labels[filename[:-4]]
        score = compare(prediction, actual)
        data.append(Transcription(filename, prediction, actual, score))

    save_results(data, output_path)


if __name__ == "__main__":
    """Script to transcribe a folder of audio"""
    parser = argparse.ArgumentParser(description="Clean & improve text for training")
    parser.add_argument("-f", "--folder", help="Audio folder", type=str, required=True)
    parser.add_argument("-l", "--labels", help="Labels path", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output text file path", type=str, required=True)
    args = parser.parse_args()

    transcribe_clips(args.folder, args.labels, args.output)
