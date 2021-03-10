import argparse
import os
import re
import logging
import json

from pydub import AudioSegment

ALHPANUMERIC = re.compile(r"\W+")


def load_audio(audio_path, sample_rate):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    return audio


def load_forced_alignment_data(forced_alignment_path):
    with open(forced_alignment_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_audio_snippet(audio, start, end, silence, output_folder, prefix=""):
    name = f"{prefix}{int(start)}_{int(end)}.wav"
    snippet = audio[start:end]
    # Pad with silence at the end
    snippet += silence
    output_path = os.path.join(output_folder, name)
    snippet.export(output_path, format="wav")
    return name


def clip_generator(
    audio_path,
    forced_alignment_path,
    output_path,
    label_path,
    logging=logging,
    min_length=1.0,
    max_length=10.0,
    silence_padding=0.1,
    sample_rate=22050,
    prefix="",
):
    os.makedirs(output_path, exist_ok=True)
    silence = AudioSegment.silent(duration=int(silence_padding * 1000))

    # Load data
    sentences = load_forced_alignment_data(forced_alignment_path)
    logging.info("Loading audio...")
    audio = load_audio(audio_path, sample_rate)
    logging.info("Loaded audio")

    # Output variables
    clip_lengths = []
    result = {}
    total = len(sentences)

    logging.info("Generating clips...")
    for i in range(len(sentences)):
        sentence = sentences[i]
        length = (sentence["end"] - sentence["start"]) // 1000

        if length >= min_length and length <= max_length:
            clip_lengths.append(length)
            name = create_audio_snippet(audio, sentence["start"], sentence["end"], silence, output_path, prefix)
            result[name] = sentence["aligned"]

    # Save text file
    with open(label_path, "w", encoding="utf-8") as f:
        for key, value in result.items():
            f.write(f"{key}|{value}\n")
    logging.info("Generated clips")

    return clip_lengths


def extend_dataset(
    audio_path,
    forced_alignment_path,
    output_path,
    label_path,
    prefix,
    logging=logging,
    min_length=1.0,
    max_length=10.0,
    silence_padding=0.1,
    sample_rate=22050,
):
    temp_label_path = "temp.csv"
    clip_generator(
        audio_path,
        forced_alignment_path,
        output_path,
        temp_label_path,
        logging,
        min_length,
        max_length,
        silence_padding,
        sample_rate,
        prefix,
    )

    with open(temp_label_path) as f:
        new_labels = f.readlines()

    with open(label_path, "a+") as f:
        for line in new_labels:
            f.write(line)

    os.remove("temp.csv")

    logging.info("Combined dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio into snippets using forced align timings")
    parser.add_argument("-a", "--audio_path", help="Path to WAV file", type=str, required=True)
    parser.add_argument(
        "-f", "--forced_alignment_path", help="Path to forced alignment JSON", type=str, default="align.json"
    )
    parser.add_argument("-o", "--output_path", help="Path to save snippets", type=str, default="wavs")
    parser.add_argument(
        "-l", "--label_path", help="Path to save snippet labelling text file", type=str, default="metadata.csv"
    )
    parser.add_argument("--min_length", help="Minumum snippet length", type=float, default=1.0)
    parser.add_argument("--max_length", help="Maximum snippet length", type=float, default=10.0)
    parser.add_argument("--silence_padding", help="Silence padding on the end of the clip", type=int, default=0.1)
    parser.add_argument("--sample_rate", help="Audio sample rate", type=int, default=22050)
    args = parser.parse_args()

    clip_lengths = clip_generator(**vars(args))

    total_audio = sum(clip_lengths)
    total_clips = len(clip_lengths)
    minutes = int(total_audio / 60)
    seconds = total_audio - (minutes * 60)
    print(f"Total clips = {total_clips} (average of {total_audio/total_clips} seconds per clip)")
    print(f"Total audio = {minutes} minutes, {seconds} seconds")
    print(f"Audio saved to {args.output_path}. Text file save to {args.label_path}")

    import matplotlib.pyplot as plt

    plt.hist(clip_lengths, bins=10)
    plt.show()
