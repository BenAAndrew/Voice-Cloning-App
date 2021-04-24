from subprocess import check_output, call, DEVNULL, STDOUT
from pathlib import Path
from pydub import AudioSegment
import os

devnull = open(os.devnull, "w")
TARGET_SAMPLE_RATE = 22050
TARGET_BITRATE = "32k"


def convert_audio(input_path):
    """
    Convert an audio file to the required format.
    This function uses FFmpeg to set the bitrate, sample rate, channels & convert to wav.

    Parameters
    ----------
    input_path : str
        Path to audio file

    Returns
    -------
    str
        Path of the converted audio
    """
    current_filename, filetype = Path(input_path).name.split(".")
    output_path = input_path.replace(current_filename, current_filename + "-converted")
    output_path = output_path.replace(filetype, "wav")
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
    current_filename = Path(input_path).name.split(".")[0]
    output_path = input_path.replace(current_filename, f"{current_filename}-{new_sample_rate}")
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
    hours = int(milliseconds / (60*60*1000))
    milliseconds = milliseconds - hours*(60*60*1000)
    minutes = int(milliseconds / (60*1000))
    milliseconds = milliseconds - minutes*(60*1000)
    seconds = int(milliseconds / 1000)
    milliseconds = milliseconds - seconds*1000
    return "%s:%s:%s.%s" % (str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2), str(milliseconds).zfill(3))


def cut_audio(input_path, start, end, output_folder):
    """
    Cuts audio to a given start & end point.

    Parameters
    ----------
    input_path : str
        Path to audio file
    start : int
        Start time in milliseconds
    end : int
        End time in milliseconds
    output_folder : str
        Folder to save audio clip to

    Returns
    -------
    str
        Path of the generated clip (named in the format 'start_end.wav')
    """
    start_timestamp = get_timestamp(start)
    duration = (end - start) / 1000
    output_name = f"{start}_{end}.wav"
    output_path = os.path.join(output_folder, output_name)
    call(["ffmpeg", "-ss", start_timestamp, "-t", str(duration), "-i", input_path, output_path], stdout=DEVNULL, stderr=STDOUT)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="audio path")
    args = parser.parse_args()
    convert_audio(args.input_path)
