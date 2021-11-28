import argparse
from subprocess import check_output, call, DEVNULL, STDOUT
from pydub import AudioSegment
import re
import os

from dataset.utils import add_suffix

devnull = open(os.devnull, "w")
TARGET_SAMPLE_RATE = 22050
TARGET_BITRATE = "32k"


def convert_audio(input_path):
    """
    Convert an audio file to the required format.
    This function uses FFmpeg to set the bitrate, sample rate, channels & convert to wav.
    Also supports extracting audio from video files.

    Parameters
    ----------
    input_path : str
        Path to audio file

    Returns
    -------
    str
        Path of the converted audio
    """
    assert os.path.isfile(input_path), f"{input_path} does not exist"
    output_path = input_path.split(".")[0] + "-converted.wav"
    assert not os.path.isfile(output_path), f"{output_path} already exists"
    check_output(
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
        ]
    )
    return output_path


def change_sample_rate(input_path, new_sample_rate):
    """
    Changes the sample rate of a given audio file.

    Parameters
    ----------
    input_path : str
        Path to audio file
    new_sample_rate : int
        Sample rate to convert audio to

    Returns
    -------
    str
        Path of the converted audio
    """
    output_path = add_suffix(input_path, str(new_sample_rate))
    check_output(["ffmpeg", "-i", input_path, "-ar", str(new_sample_rate), output_path])
    return output_path


def get_timestamp(milliseconds):
    """
    Generates timestamp for an amount of milliseconds

    Parameters
    ----------
    milliseconds : int
        Time in milliseconds

    Returns
    -------
    str
        Timestamp (in format H:M:S.milli)
    """
    hours = int(milliseconds / (60 * 60 * 1000))
    milliseconds = milliseconds - hours * (60 * 60 * 1000)
    minutes = int(milliseconds / (60 * 1000))
    milliseconds = milliseconds - minutes * (60 * 1000)
    seconds = int(milliseconds / 1000)
    milliseconds = milliseconds - seconds * 1000
    return "%s:%s:%s.%s" % (
        str(hours).zfill(2),
        str(minutes).zfill(2),
        str(seconds).zfill(2),
        str(milliseconds).zfill(3),
    )


def cut_audio(input_path, start, end, output_folder):
    """
    Cuts audio to a given start & end timestamp.

    Parameters
    ----------
    input_path : str
        Path to audio file
    start : int
        Start timestamp (H:M:S.milli)
    end : int
        End timestamp (H:M:S.milli)
    output_folder : str
        Folder to save audio clip to

    Returns
    -------
    str
        Path of the generated clip
    """

    def _timestamp_to_filename(timestamp):
        """Removes non-numeric characters from timestamp"""
        return re.sub("[^0-9]", "", timestamp)

    assert os.path.isfile(input_path), f"{input_path} does not exist"
    output_name = f"{_timestamp_to_filename(start)}_{_timestamp_to_filename(end)}.wav"
    output_path = os.path.join(output_folder, output_name)
    assert not os.path.isfile(output_path), f"{output_path} already exists"
    call(
        ["ffmpeg", "-ss", start, "-to", end, "-i", input_path, output_path],
        stdout=DEVNULL,
        stderr=STDOUT,
    )
    return output_name


def add_silence(input_path, silence):
    """
    Adds silence to the end of a clip.
    This is needed as Tacotron2 sometimes has alignment issues if speech
    continues right until the end of the clip.

    Parameters
    ----------
    input_path : str
        Path to audio file (overwrites to add silence)
    silence : AudioSegment
        Pydub audiosegement of silence
    """
    audio = AudioSegment.from_file(input_path)
    audio += silence
    audio.export(input_path, format="wav")


if __name__ == "__main__":
    """Audio conversion enabled from CLI"""
    parser = argparse.ArgumentParser(description="Convert audio to required format")
    parser.add_argument("-i", "--input_path", type=str, help="audio path")
    args = parser.parse_args()
    output_path = convert_audio(args.input_path)
    print("Converted audio saved to ", output_path)
