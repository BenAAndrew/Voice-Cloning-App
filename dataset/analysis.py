import argparse
from datetime import timedelta
import os
import re

from pydub import AudioSegment

ALLOWED_CHARACTERS_RE = re.compile("[^a-zA-Z ]+")


def get_text(metadata_file):
    text = []
    with open(metadata_file, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(ALLOWED_CHARACTERS_RE, "", line.split("|")[1].lower().strip())
            for word in line.split(" "):
                text.append(word.strip())
    return text


def get_clip_lengths(metadata_file):
    durations = []
    with open(metadata_file, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            text = line.split(".wav")[0]
            start, end = text.split("_")
            duration = int(end) - int(start)
            durations.append(duration)
    return durations


def get_total_audio_duration(metadata_file):
    clip_lengths = get_clip_lengths(metadata_file)
    return int(sum(clip_lengths) / 1000), len(clip_lengths)


if __name__ == "__main__":
    """ Script to analyse dataset """
    parser = argparse.ArgumentParser(description="Analyse dataset")
    parser.add_argument("-m", "--metadata", help="Path to metadata file", type=str, required=True)
    args = parser.parse_args()

    clip_lengths = get_clip_lengths(args.metadata)
    text = get_text(args.metadata)
    duration = sum(clip_lengths) / 1000

    print(f"Total clips: {len(clip_lengths)}")
    print(f"Total words: {len(text)}")
    print(f"Total distinct words: {len(set(text))}")
    print(f"Mean words per clip: {len(text)/len(clip_lengths)}")

    print(f"Total duration: {timedelta(seconds=duration)}")
    print(f"Mean clip duration {duration/len(clip_lengths)}")
    print(f"Min clip duration {min(clip_lengths)/1000}")
    print(f"Max clip duration {max(clip_lengths)/1000}")
