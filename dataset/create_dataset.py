import argparse
import logging
from os.path import dirname, abspath
import shutil
import sys
import os


sys.path.append(dirname(dirname(abspath(__file__))))
from training import DEFAULT_ALPHABET
from training.utils import load_symbols
from dataset import AUDIO_FOLDER, UNLABELLED_FOLDER, METADATA_FILE, ALIGNMENT_FILE, INFO_FILE, MIN_LENGTH, MAX_LENGTH
from dataset.audio_processing import convert_audio
from dataset.clip_generator import clip_generator
from dataset.analysis import save_dataset_info, get_text
from dataset.transcribe import Silero


def create_dataset(
    text_path,
    audio_path,
    transcription_model,
    output_folder,
    logging=logging,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH,
    min_confidence=0.85,
    combine_clips=True,
    symbols=DEFAULT_ALPHABET,
):
    """
    Generates a dataset.
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
    logging.info(f"Converting {audio_path}...")
    converted_audio = convert_audio(audio_path)
    forced_alignment_path = os.path.join(output_folder, ALIGNMENT_FILE)
    output_path = os.path.join(output_folder, AUDIO_FOLDER)
    unlabelled_path = os.path.join(output_folder, UNLABELLED_FOLDER)
    label_path = os.path.join(output_folder, METADATA_FILE)
    info_path = os.path.join(output_folder, INFO_FILE)

    try:
        clip_lengths = clip_generator(
            converted_audio,
            text_path,
            transcription_model,
            forced_alignment_path,
            output_path,
            unlabelled_path,
            label_path,
            logging=logging,
            symbols=symbols,
            min_length=min_length,
            max_length=max_length,
            min_confidence=min_confidence,
            combine_clips=combine_clips,
        )
    except Exception as e:
        shutil.rmtree(output_folder)
        raise e

    logging.info("Getting dataset info...")
    save_dataset_info(get_text(label_path), output_path, info_path, clip_lengths)


if __name__ == "__main__":
    """Generate dataset"""
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("-t", "--text_path", help="Path to text file", type=str, required=True)
    parser.add_argument("-a", "--audio_path", help="Path to audio file", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Path to save snippets", type=str, default="wavs")
    parser.add_argument("-l", "--language", help="The language to use", type=str, default="English")
    parser.add_argument("-s", "--symbol_path", help="Path to symbol/alphabet file", type=str, default=None)
    args = parser.parse_args()

    create_dataset(
        text_path=args.text_path,
        audio_path=args.audio_path,
        transcription_model=Silero(args.language),
        output_folder=args.output_folder,
        symbols=load_symbols(args.symbol_path) if args.symbol_path else DEFAULT_ALPHABET,
    )
