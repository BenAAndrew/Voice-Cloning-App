import subprocess
import argparse
from pathlib import Path
import os

TARGET_SAMPLE_RATE = 22050
TARGET_BITRATE = "32k"


def compress_audio(input_path): 
    current_filename = Path(input_path).name.split(".")[0]
    output_path = input_path.replace(current_filename, current_filename+"-compressed")
    
    command = ["ffmpeg", "-i", input_path, "-b:a", str(TARGET_BITRATE), "-ac", "1", "-map", "a", "-ar", str(TARGET_SAMPLE_RATE), output_path]
    subprocess.check_output(command)

    os.remove(input_path)
    os.rename(output_path, input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="audio path")
    args = parser.parse_args()

    compress_audio(args.input_path)
