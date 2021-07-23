import argparse
import os
import re
import librosa
import json

from dataset.clip_generator import CHARACTER_ENCODING


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
        data = json.load(f)
        return data["total_duration"], data["total_clips"]


def save_dataset_info(metadata_file, folder, output_path, clip_lengths=None):
    """
    Save dataset properties to info JSON.

    Parameters
    ----------
    metadata_file : str
        Path to metadata file
    folder : str
        Path to audio folder
    output_path : str
        Path to save info JSON to
    clip_lengths : list (optional)
        List of clip lengths
    """
    if not clip_lengths:
        clip_lengths = get_clip_lengths(folder)
    words = get_text(metadata_file)
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


def validate_dataset(folder, metadata_file="metadata.csv", audio_folder="wavs", info_file="info.json"):
    """
    Validate a dataset has all required files.

    Parameters
    ----------
    folder : str
        Path to dataset folder
    metadata_file : str
        Metadata file name
    audio_folder : str
        Audio folder name
    info_file : str
        Info file name

    Returns
    -------
    str
        Error message or None if no error is produced
    """
    if not os.path.isfile(os.path.join(folder, metadata_file)):
        return f"Missing {metadata_file} file"
    if not os.path.isfile(os.path.join(folder, info_file)):
        return f"Missing {info_file} file"
    if not os.path.isdir(os.path.join(folder, audio_folder)):
        return f"Missing {audio_folder} folder"
    return None


if __name__ == "__main__":
    """Script to analyse dataset"""
    parser = argparse.ArgumentParser(description="Analyse dataset")
    parser.add_argument("-w", "--wavs", help="Path to wavs folder", type=str, required=True)
    parser.add_argument("-m", "--metadata", help="Path to metadata file", type=str, required=True)
    parser.add_argument("-o", "--output_path", help="Path to save JSON file", type=str, required=True)
    args = parser.parse_args()

    save_dataset_info(args.metadata, args.wavs, args.output_path)

    with open(args.output_path) as f:
        data = json.load(f)
        for key, value in data.items():
            print(key, ":", value)
