import argparse
import logging
import os
from os.path import dirname, abspath
import sys
import shutil
from pathlib import Path

sys.path.append(dirname(dirname(abspath(__file__))))

from training import DEFAULT_ALPHABET
from training.utils import load_symbols
from dataset.audio_processing import convert_audio
from dataset.clip_generator import clip_generator
from dataset.analysis import save_dataset_info, get_text
from dataset.transcribe import Silero
from dataset import AUDIO_FOLDER, UNLABELLED_FOLDER, METADATA_FILE, ALIGNMENT_FILE, INFO_FILE, MIN_LENGTH, MAX_LENGTH
from dataset.utils import add_suffix


def extend_existing_dataset(
    text_path,
    audio_path,
    transcription_model,
    output_folder,
    suffix,
    logging=logging,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH,
    min_confidence=0.85,
    combine_clips=True,
    symbols=DEFAULT_ALPHABET,
):
    """
    Extends an existing dataset.
    Converts audio to required format, generates clips & produces required files.

    Parameters
    ----------
    text_path : str
        Path to source text
    audio_path : str
        Path to source audio
    transcription_model : TranscriptionModel
        Transcription model
    output_folder : str
        Path to save dataset to
    suffix : str
        String suffix to append to filenames
    logging : logging (optional)
        Logging object to write logs to
    min_length : float (optional)
        Minimum duration of a clip in seconds
    max_length : float (optional)
        Maximum duration of a clip in seconds
    min_confidence : float (optional)
        Minimum confidence score to generate a clip for
    combine_clips : bool (optional)
        Whether to combine clips to make them longer
    symbols : list[str] (optional)
        list of valid symbols default to DEFAULT_ALPHABET

    Raises
    -------
    AssertionError
        If given paths are invalid or clips could not be produced
    """
    assert os.path.isdir(output_folder), "Missing existing dataset clips folder"
    logging.info(f"Converting {audio_path}...")
    converted_audio = convert_audio(audio_path)

    forced_alignment_path = os.path.join(output_folder, ALIGNMENT_FILE)
    output_path = os.path.join(output_folder, AUDIO_FOLDER)
    unlabelled_path = os.path.join(output_folder, UNLABELLED_FOLDER)
    label_path = os.path.join(output_folder, METADATA_FILE)
    info_path = os.path.join(output_folder, INFO_FILE)
    temp_label_path = label_path.replace(Path(label_path).name, "temp.csv")
    temp_unlabelled_folder = unlabelled_path.replace(Path(unlabelled_path).name, "temp_unlabelled")
    temp_wavs_folder = output_path.replace(Path(output_path).name, "temp_wavs")
    os.makedirs(unlabelled_path, exist_ok=True)

    clip_generator(
        converted_audio,
        text_path,
        transcription_model,
        forced_alignment_path,
        temp_wavs_folder,
        temp_unlabelled_folder,
        temp_label_path,
        logging=logging,
        symbols=symbols,
        min_length=min_length,
        max_length=max_length,
        min_confidence=min_confidence,
        combine_clips=combine_clips,
    )

    with open(temp_label_path) as f:
        new_labels = f.readlines()

    with open(label_path, "a+") as f:
        for line in new_labels:
            filename, text = line.split("|")
            new_filename = add_suffix(filename, suffix)
            f.write(f"{new_filename}|{text}")

    for filename in os.listdir(temp_wavs_folder):
        new_filename = add_suffix(filename, suffix)
        shutil.copyfile(os.path.join(temp_wavs_folder, filename), os.path.join(output_path, new_filename))

    for filename in os.listdir(temp_unlabelled_folder):
        new_filename = add_suffix(filename, suffix)
        shutil.copyfile(os.path.join(temp_unlabelled_folder, filename), os.path.join(unlabelled_path, new_filename))

    os.remove(temp_label_path)
    shutil.rmtree(temp_wavs_folder)
    shutil.rmtree(temp_unlabelled_folder)
    logging.info("Combined dataset")

    logging.info("Getting dataset info...")
    # Do not pass clip lengths from extend_dataset as we need to get size of entire dataset (not just new clips)
    save_dataset_info(get_text(label_path), output_path, info_path)


if __name__ == "__main__":
    """Extend existing dataset"""
    parser = argparse.ArgumentParser(description="Extend existing dataset")
    parser.add_argument("-t", "--text_path", help="Path to text file", type=str, required=True)
    parser.add_argument("-a", "--audio_path", help="Path to audio file", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Path to save dataset", type=str, default="wavs")
    parser.add_argument("-s", "--suffix", help="String suffix for added files", type=str, required=True)
    parser.add_argument("-l", "--language", help="The language to use", type=str, default="English")
    parser.add_argument("-s", "--symbol_path", help="Path to symbol/alphabet file", type=str, default=None)
    args = parser.parse_args()

    extend_existing_dataset(
        text_path=args.text_path,
        audio_path=args.audio_path,
        transcription_model=Silero(args.language),
        output_folder=args.output_folder,
        suffix=args.suffix,
        symbols=load_symbols(args.symbol_path) if args.symbol_path else DEFAULT_ALPHABET,
    )
