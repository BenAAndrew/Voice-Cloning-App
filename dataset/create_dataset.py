import argparse
import logging
from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

from dataset.audio_processing import convert_audio
from dataset.clip_generator import clip_generator, MIN_LENGTH, MAX_LENGTH
from dataset.analysis import save_dataset_info
from dataset.transcribe import create_transcription_model


def create_dataset(
    text_path,
    audio_path,
    transcription_model,
    forced_alignment_path,
    output_path,
    label_path,
    info_path,
    logging=logging,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH,
    min_confidence=0.85,
    combine_clips=True,
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
    forced_alignment_path : str
        Path to save alignment JSON to
    output_path : str
        Path to save audio clips to
    label_path : str
        Path to save label file to
    info_path : str
        Path to save info JSON to
    logging : logging (optional)
        Logging object to write logs to
    min_confidence : float (optional)
        Minimum confidence score to generate a clip for

    Raises
    -------
    AssertionError
        If given paths are invalid or clips could not be produced
    """
    logging.info(f"Coverting {audio_path}...")
    converted_audio = convert_audio(audio_path)
    clip_lengths = clip_generator(
        converted_audio,
        text_path,
        transcription_model,
        forced_alignment_path,
        output_path,
        label_path,
        logging=logging,
        min_length=min_length,
        max_length=max_length,
        min_confidence=min_confidence,
        combine_clips=combine_clips,
    )
    logging.info("Getting dataset info...")
    save_dataset_info(label_path, output_path, info_path, clip_lengths)


if __name__ == "__main__":
    """Generate dataset"""
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("-t", "--text_path", help="Path to text file", type=str, required=True)
    parser.add_argument("-a", "--audio_path", help="Path to audio file", type=str, required=True)
    parser.add_argument(
        "-f", "--forced_alignment_path", help="Path to forced alignment JSON", type=str, default="align.json"
    )
    parser.add_argument("-o", "--output_path", help="Path to save snippets", type=str, default="wavs")
    parser.add_argument(
        "-l", "--label_path", help="Path to save snippet labelling text file", type=str, default="metadata.csv"
    )
    parser.add_argument("-i", "--info_path", help="Path to save info file", type=str, default="info.json")
    args = parser.parse_args()

    create_dataset(**vars(args), transcription_model=create_transcription_model())
