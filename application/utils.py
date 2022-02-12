import logging
import os
import io
from datetime import datetime
import traceback
import shutil
import zipfile
import librosa
from flask import send_file
import resampy  # noqa

from main import socketio
from dataset import CHARACTER_ENCODING
from dataset.audio_processing import convert_audio
from dataset.analysis import save_dataset_info, get_text


class SocketIOHandler(logging.Handler):
    """
    Sends logger messages to the frontend using flask-socketio.
    These are handled in application.js
    """

    def emit(self, record):
        text = record.getMessage()
        if text.startswith("Progress"):
            text = text.split("-")[1]
            current, total = text.split("/")
            socketio.emit("progress", {"number": current, "total": total}, namespace="/voice")
        elif text.startswith("Status"):
            socketio.emit("status", {"text": text.replace("Status -", "")}, namespace="/voice")
        elif text.startswith("Alignment"):
            text = text.split("- ")[1]
            iteration, image = text.split(", ")
            socketio.emit("alignment", {"iteration": iteration, "image": image}, namespace="/voice")
        else:
            socketio.emit("logs", {"text": text}, namespace="/voice")


# Data
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice")
logger.addHandler(SocketIOHandler())
thread = None


def background_task(func, **kwargs):
    """
    Runs a background task.
    If function errors out it will send an error log to the error logging server and page.
    Sends 'done' message to frontend when complete.

    Parameters
    ----------
    func : function
        Function to run in background
    kwargs : kwargs
        Kwargs to pass to function
    """
    try:
        socketio.sleep(5)
        func(logging=logger, **kwargs)
    except Exception as e:
        error = {"type": e.__class__.__name__, "text": str(e), "stacktrace": traceback.format_exc()}
        socketio.emit("error", error, namespace="/voice")
        raise e

    socketio.emit("done", {"text": None}, namespace="/voice")


def start_progress_thread(func, **kwargs):
    """
    Starts a background task using socketio.

    Parameters
    ----------
    func : function
        Function to run in background
    kwargs : kwargs
        Kwargs to pass to function
    """
    global thread
    print("Starting Thread")
    thread = socketio.start_background_task(background_task, func=func, **kwargs)


def serve_file(path, filename, mimetype, as_attachment=True):
    """
    Serves a file as a response

    Parameters
    ----------
    path : str
        Path to file
    filename : str
        Filename of generated attachment
    mimetype : str
        Mimetype of file
    as_attachment : bool (optional)
        Whether to respond as an attachment for download (default is True)
    """
    with open(path, "rb") as f:
        return send_file(
            io.BytesIO(f.read()), attachment_filename=filename, mimetype=mimetype, as_attachment=as_attachment
        )


def get_next_url(urls, path):
    """
    Returns the URL of the next step in the voice cloning process.

    Parameters
    ----------
    urls : dict
        Frontend url paths and names
    path : str
        Current URL

    Returns
    -------
    str
        URL of next step or '' if not found
    """
    urls = list(urls.keys())
    next_url_index = urls.index(path) + 1
    return urls[next_url_index] if next_url_index < len(urls) else ""


def get_suffix():
    """
    Generates a filename suffix using the currrent datetime.

    Returns
    -------
    str
        String suffix
    """
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def delete_folder(path):
    """
    Deletes a folder.

    Parameters
    ----------
    path : str
        Path to folder

    Raises
    -------
    AssertionError
        If folder is not found
    """
    assert os.path.isdir(path), f"{path} does not exist"
    shutil.rmtree(path)


def import_dataset(dataset, dataset_directory, audio_folder, logging):
    """
    Imports a dataset zip into the app.
    Checks required files are present, saves the files,
    converts the audio to the required format and generates the info file.
    Deletes given zip regardless of success.

    Parameters
    ----------
    dataset : str
        Path to dataset zip
    dataset_directory : str
        Destination path for the dataset
    audio_folder : str
        Destination path for the dataset audio
    logging : logging
        Logging object to write logs to

    Raises
    -------
    AssertionError
        If files are missing or invalid
    """
    try:
        with zipfile.ZipFile(dataset, mode="r") as z:
            files_list = z.namelist()

            assert ("metadata.csv" in files_list) or (
                "trainlist.txt" in files_list and "vallist.txt" in files_list
            ), "Dataset doesn't include metadata.csv or trainlist.txt/vallist.txt. Make sure this is in the root of the zip file"

            folders = [x.split("/")[0] for x in files_list if "/" in x]
            assert (
                "wavs" in folders
            ), "Dataset missing wavs folder. Make sure this folder is in the root of the zip file"

            wavs = [x for x in files_list if x.startswith("wavs/") and x.endswith(".wav")]
            assert wavs, "No wavs found in wavs folder"

            logging.info("Creating directory")
            os.makedirs(dataset_directory, exist_ok=False)
            os.makedirs(audio_folder, exist_ok=False)

            if "metadata.csv" in files_list:
                metadata = z.read("metadata.csv").decode(CHARACTER_ENCODING, "ignore").replace("\r\n", "\n")
                num_metadata_rows = len([row for row in metadata.split("\n") if row])
                assert (
                    len(wavs) == num_metadata_rows
                ), f"Number of wavs and labels do not match. metadata: {num_metadata_rows}, wavs: {len(wavs)}"

                # Save metadata
                logging.info("Saving files")
                with open(os.path.join(dataset_directory, "metadata.csv"), "w", encoding=CHARACTER_ENCODING) as f:
                    f.write(metadata)
            else:
                trainlist = z.read("trainlist.txt").decode(CHARACTER_ENCODING, "ignore").replace("\r\n", "\n")
                vallist = z.read("vallist.txt").decode(CHARACTER_ENCODING, "ignore").replace("\r\n", "\n")
                num_rows = len([row for row in trainlist.split("\n") if row]) + len(
                    [row for row in vallist.split("\n") if row]
                )
                assert (
                    len(wavs) == num_rows
                ), f"Number of wavs and labels do not match. trainlist+vallist: {num_rows}, wavs: {len(wavs)}"

                # Save trainlist & vallist
                logging.info("Saving files")
                with open(os.path.join(dataset_directory, "trainlist.txt"), "w", encoding=CHARACTER_ENCODING) as f:
                    f.write(trainlist)
                with open(os.path.join(dataset_directory, "vallist.txt"), "w", encoding=CHARACTER_ENCODING) as f:
                    f.write(vallist)

            # Save wavs
            total_wavs = len(wavs)
            clip_lengths = []
            filenames = {}
            for i in range(total_wavs):
                wav = wavs[i]
                data = z.read(wav)
                path = os.path.join(dataset_directory, "wavs", wav.split("/")[1])
                with open(path, "wb") as f:
                    f.write(data)
                    new_path = convert_audio(path)
                    duration = librosa.get_duration(filename=new_path)
                    clip_lengths.append(duration)
                    filenames[path] = new_path
                logging.info(f"Progress - {i+1}/{total_wavs}")

            logging.info(f"Longest clip: {max(clip_lengths)}s, Shortest clip: {min(clip_lengths)}s")
            # Get around "file in use" by using delay
            logging.info("Deleting temp files")
            for old_path, new_path in filenames.items():
                os.remove(old_path)
                os.rename(new_path, old_path)

            # Create info file
            logging.info("Creating info file")
            words = (
                get_text(os.path.join(dataset_directory, "metadata.csv"))
                if "metadata.csv" in files_list
                else get_text(os.path.join(dataset_directory, "trainlist.txt"))
                + get_text(os.path.join(dataset_directory, "vallist.txt"))
            )
            save_dataset_info(
                words,
                os.path.join(dataset_directory, "wavs"),
                os.path.join(dataset_directory, "info.json"),
                clip_lengths=clip_lengths,
            )
    except Exception as e:
        os.remove(dataset)
        raise e

    os.remove(dataset)
