import argparse
import os
import re
import logging

from pydub import AudioSegment
from tqdm import tqdm

ALHPANUMERIC = re.compile(r"\W+")


class Label:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


def load_audio(audio_path, sample_rate):
    if audio_path.endswith("wav"):
        audio = AudioSegment.from_wav(audio_path)
    else:
        audio = AudioSegment.from_mp3(audio_path)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    return audio


def load_text_file_words(text_file_path):
    with open(text_file_path) as f:
        text = f.read()
        text = text.replace("\n", " ")
        text = text.replace("-", " ")
        text = text.replace("...", "... ")
        return [item for item in text.split(" ") if re.sub(ALHPANUMERIC, "", item)]


def load_forced_alignment_data(forced_alignment_path, text):
    successfully_aligned_words = 0
    sections = []
    section = []

    with open(forced_alignment_path, "r", encoding="utf-8") as f:
        rows = f.readlines()
        for i in range(len(rows)):
            row = rows[i]
            word, guess, start, end = row.strip().split(",")
            assert re.sub(ALHPANUMERIC, "", word) == re.sub(
                ALHPANUMERIC, "", text[i]
            ), f"{word} != {text[i]} on line {i+1}"
            if word.lower() == guess.lower():
                successfully_aligned_words += 1
                section.append(Label(text[i], float(start), float(end)))
            elif section:
                sections.append(section)
                section = []

    if section:
        sections.append(section)
    print(f"{successfully_aligned_words}/{len(rows)} successfully aligned")
    return sections


def create_audio_snippet(audio, start, end, silence, output_folder):
    # Convert to ms
    name = f"{start}_{end}.wav"
    snippet = audio[start:end]
    # Pad with silence at the end
    snippet += silence
    # Save
    output_path = os.path.join(output_folder, name)
    snippet.export(output_path, format="wav")
    return name


def clip_generator(
    audio_path,
    text_path,
    forced_alignment_path,
    output_path,
    label_path,
    logging=logging,
    min_length=1.0,
    max_length=10.0,
    gap=0.7,
    next_word_padding=0.0,
    silence_padding=0.1,
    sample_rate=22050,
):
    os.makedirs(output_path, exist_ok=True)
    silence = AudioSegment.silent(duration=int(silence_padding * 1000))

    # Load data
    text = load_text_file_words(text_path)
    sections = load_forced_alignment_data(forced_alignment_path, text)

    logging.info("Loading audio...")
    audio = load_audio(audio_path, sample_rate)
    logging.info("Loaded audio")

    # Output variables
    clip_lengths = []
    result = {}

    total = len(sections)
    counter = 0

    for section in sections:
        sentence = []

        for i in range(len(section)):
            label = section[i]
            sentence.append(label)
            start = sentence[0].start
            length = sentence[-1].end - start
            gap_between_last_word = sentence[-1].start - sentence[-2].end if len(sentence) > 1 else 0

            # End snippet if last word, there was a large gap or it's getting too long
            if i == len(section) - 1 or gap_between_last_word > gap or length >= max_length:
                # Cutoff last word
                if gap_between_last_word > gap or length >= max_length:
                    sentence = sentence[:-1]
                    length = sentence[-1].end - start

                if length >= min_length:
                    # Save snippet
                    clip_lengths.append(length)
                    start = int(start * 1000)
                    end = int(sentence[-1].end * 1000) + int(next_word_padding * 1000)
                    name = create_audio_snippet(audio, start, end, silence, output_path)
                    result[name] = " ".join([l.word for l in sentence])

                # Reset running variable
                sentence = [label]

        counter += 1
        logging.info(f"Progress - {counter}/{total}")

    # Save text file
    with open(label_path, "w", encoding="utf-8") as f:
        for key, value in result.items():
            f.write(f"{key}|{value}\n")

    return clip_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio into snippets using forced align timings")
    parser.add_argument("-a", "--audio_path", help="Path to WAV file", type=str, required=True)
    parser.add_argument("-t", "--text_path", help="Path to source text file", type=str, required=True)
    parser.add_argument(
        "-f", "--forced_alignment_path", help="Path to forced alignment CSV", type=str, default="align.csv"
    )
    parser.add_argument("-o", "--output_path", help="Path to save snippets", type=str, default="wavs")
    parser.add_argument(
        "-l", "--label_path", help="Path to save snippet labelling text file", type=str, default="metadata.csv"
    )
    parser.add_argument("--min_length", help="Minumum snippet length", type=float, default=1.0)
    parser.add_argument("--max_length", help="Maximum snippet length", type=float, default=10.0)
    parser.add_argument("--gap", help="Maximum gap between words for snippet to end", type=float, default=0.7)
    parser.add_argument(
        "--next_word_padding", help="Additional gap from end of this word to next", type=float, default=0.0
    )
    parser.add_argument("--silence_padding", help="Silence padding on the end of the clip", type=int, default=0.1)
    parser.add_argument("--sample_rate", help="Audio sample rate", type=int, default=22050)
    args = parser.parse_args()

    clip_lengths = clip_generator(**args)

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
