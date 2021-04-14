import argparse
import logging
import os
from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

from dataset.audio_processing import convert_audio
from dataset.clip_generator import extend_dataset
from dataset.analysis import save_dataset_info


def extend_existing_dataset(
    text_path, audio_path, forced_alignment_path, output_path, label_path, suffix, info_path, logging=logging
):
    assert os.path.isdir(output_path), "Missing existing dataset clips folder"
    assert os.path.isfile(label_path), "Missing existing dataset metadata file"
    logging.info(f"Coverting {audio_path}...")
    converted_audio = convert_audio(audio_path)
    extend_dataset(converted_audio, text_path, forced_alignment_path, output_path, label_path, suffix, logging=logging)
    logging.info("Getting dataset info...")
    save_dataset_info(label_path, output_path, info_path)


if __name__ == "__main__":
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
    parser.add_argument("-s", "--suffix", help="String suffix for added files", type=str, required=True)
    parser.add_argument("-i", "--info_path", help="Path to save info file", type=str, default="info.json")
    args = parser.parse_args()

    extend_existing_dataset(**vars(args))
