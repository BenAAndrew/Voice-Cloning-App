from subprocess import call
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
import os

devnull = open(os.devnull, "w")
TARGET_SAMPLE_RATE = 22050
TARGET_BITRATE = "32k"


def convert_audio(input_path):
    current_filename, filetype = Path(input_path).name.split(".")
    output_path = input_path.replace(current_filename, current_filename + "-converted")
    output_path = output_path.replace(filetype, "wav")
    call(
        [
            "ffmpeg",
            "-i",
            input_path,
            "-b:a",
            TARGET_BITRATE,
            "-ac",
            "1",
            "-map",
            "a",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            output_path,
        ],
        stdout=devnull,
        stderr=devnull,
    )
    return output_path


def change_sample_rate(input_path, new_sample_rate):
    current_filename = Path(input_path).name.split(".")[0]
    output_path = input_path.replace(current_filename, f"{current_filename}-{new_sample_rate}")
    call(["ffmpeg", "-i", input_path, "-ar", str(new_sample_rate), output_path], stdout=devnull, stderr=devnull)
    return output_path


def cut_audio(input_path, start, end, output_folder):
    start_timestamp = datetime.fromtimestamp(start / 1000).strftime("%H:%M:%S.%f")
    duration = (end - start) / 1000
    output_name = f"{start}_{end}.wav"
    output_path = os.path.join(output_folder, output_name)
    call(
        ["ffmpeg", "-ss", start_timestamp, "-t", str(duration), "-i", input_path, output_path],
        stdout=devnull,
        stderr=devnull,
    )
    return output_name


def add_silence(input_path, silence):
    audio = AudioSegment.from_file(input_path)
    audio += silence
    audio.export(input_path, format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="audio path")
    args = parser.parse_args()
    convert_audio(args.input_path)
