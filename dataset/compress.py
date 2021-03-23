from subprocess import call
import argparse

TARGET_SAMPLE_RATE = 22050
TARGET_BITRATE = "32k"


def compress_audio(input_path, output_path):
    command = (
        f"ffmpeg -i {input_path} -acodec libmp3lame -b:a {TARGET_BITRATE} -ac 1 -ar {TARGET_SAMPLE_RATE} {output_path}"
    )
    call(command.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input audio path")
    parser.add_argument("-o", "--output_path", type=str, help="output audio path")
    args = parser.parse_args()

    compress_audio(args.input_path, args.output_path)
