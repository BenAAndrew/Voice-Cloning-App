import re
import os

REGEX = re.compile("[^a-zA-Z ]")
TOTAL_SAMPLES = 100


def load_file(path):
    with open(path, encoding="utf-8") as f:
        data = {}
        for line in f.readlines():
            item = line.split("|")
            data[item[0]] = item[1]
        return data


def compare(expected_text, actual_text):
    expected_text = REGEX.sub("", expected_text.strip()).lower()
    actual_text = REGEX.sub("", actual_text.strip()).lower()
    expected_words = expected_text.split(" ")
    actual_words = actual_text.split(" ")
    difference = set(expected_words) - set(actual_words)
    return 1 - (len(difference) / len(expected_words))


def score_transcription(aligned_file, predicted_file):
    aligned = load_file(aligned_file)
    predicted = load_file(predicted_file)
    scores = []

    for filename, text in predicted.items():
        scores.append(compare(text, aligned[filename]))

    return sum(scores) / len(scores)
