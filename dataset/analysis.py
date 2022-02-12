import argparse
import os
import re
import librosa
import json

from dataset import CHARACTER_ENCODING, AUDIO_FOLDER, METADATA_FILE, INFO_FILE
from training import TRAIN_FILE, VALIDATION_FILE


ALLOWED_CHARACTERS_RE = re.compile("[^a-zA-Z ]+")


def get_text(metadata_file):
    """
    Get all words in a metadata file.

    Parameters
    ----------
    metadata_file : str
        Path to metadata file

    Returns
    -------
    list
        All words in text file
    """
    text = []
    with open(metadata_file, encoding=CHARACTER_ENCODING) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub(ALLOWED_CHARACTERS_RE, "", line.split("|")[1].lower().strip())
            for word in line.split(" "):
                text.append(word.strip())
    return text


def get_clip_lengths(folder):
    """
    Get duration of all clips in a given folder.

    Parameters
    ----------
    folder : str
        Path to clip folder

    Returns
    -------
    list
        Lengths in seconds of clips in folder
    """
    return [librosa.get_duration(filename=os.path.join(folder, filename)) for filename in os.listdir(folder)]


def get_total_audio_duration(info_file):
    """
    Get duration and total clips from info JSON.

    Parameters
    ----------
    info_file : str
        Path to info JSON

    Returns
    -------
    float
        Total duration of all clips
    int
        Total number of clips
    """
    with open(info_file) as f:
        return json.load(f)


def save_dataset_info(words, folder, output_path, clip_lengths=None):
    """
    Save dataset properties to info JSON.

    Parameters
    ----------
    words : list
        List of words in text
    folder : str
        Path to audio folder
    output_path : str
        Path to save info JSON to
    clip_lengths : list (optional)
        List of clip lengths
    """
    if not clip_lengths:
        clip_lengths = get_clip_lengths(folder)
    total_duration = sum(clip_lengths)
    total_words = len(words)
    total_clips = len(clip_lengths)

    data = {
        "total_duration": total_duration,
        "total_clips": len(clip_lengths),
        "mean_clip_duration": total_duration / len(clip_lengths),
        "max_clip_duration": max(clip_lengths),
        "min_clip_duration": min(clip_lengths),
        "total_words": total_words,
        "total_distinct_words": len(set(words)),
        "mean_words_per_clip": total_words / total_clips,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def update_dataset_info(metadata_file, json_file, clip_path, text):
    """
    Extend dataset info JSON with additional clip

    Parameters
    ----------
    metadata_file : str
        Path to metadata file
    json_file : str
        Path to the dataset info JSON
    clip_path : str
        Path to the audio clip
    text : str
        Text in clip
    """
    assert os.path.isfile(metadata_file)
    assert os.path.isfile(json_file)
    assert os.path.isfile(clip_path)

    with open(json_file) as f:
        data = json.load(f)

    clip_length = librosa.get_duration(filename=clip_path)
    words = get_text(metadata_file)
    clip_words = text.split(" ")
    all_words = words + clip_words

    data["total_duration"] += clip_length
    data["total_clips"] += 1
    data["mean_clip_duration"] == data["total_duration"] / data["total_clips"]
    data["max_clip_duration"] = max(data["max_clip_duration"], clip_length)
    data["min_clip_duration"] = min(data["min_clip_duration"], clip_length)
    data["total_words"] += len(clip_words)
    data["total_distinct_words"] = len(set(all_words))
    data["mean_words_per_clip"] = data["total_words"] / data["total_clips"]

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


def validate_dataset(folder):
    """
    Validate a dataset has all required files.

    Parameters
    ----------
    folder : str
        Path to dataset folder

    Returns
    -------
    str
        Error message or None if no error is produced
    """
    if not os.path.isfile(os.path.join(folder, METADATA_FILE)) and not (
        os.path.isfile(os.path.join(folder, TRAIN_FILE)) and os.path.isfile(os.path.join(folder, VALIDATION_FILE))
    ):
        return f"Missing {METADATA_FILE} or {TRAIN_FILE}/{VALIDATION_FILE} file"
    if not os.path.isfile(os.path.join(folder, INFO_FILE)):
        return f"Missing {INFO_FILE} file"
    if not os.path.isdir(os.path.join(folder, AUDIO_FOLDER)):
        return f"Missing {AUDIO_FOLDER} folder"
    return None


if __name__ == "__main__":
    """Script to analyse dataset"""
    parser = argparse.ArgumentParser(description="Analyse dataset")
    parser.add_argument("-w", "--wavs", help="Path to wavs folder", type=str, required=True)
    parser.add_argument("-m", "--metadata", help="Path to metadata file", type=str, required=True)
    parser.add_argument("-o", "--output_path", help="Path to save JSON file", type=str, required=True)
    args = parser.parse_args()

    save_dataset_info(get_text(args.metadata), args.wavs, args.output_path)

    with open(args.output_path) as f:
        data = json.load(f)
        for key, value in data.items():
            print(key, ":", value)
