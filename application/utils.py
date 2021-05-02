import logging
from threading import Thread
import os
from datetime import datetime
import requests
import traceback
import configparser
import shutil
import zipfile
import librosa
import time

from main import socketio
from dataset.audio_processing import convert_audio
from dataset.clip_generator import clip_generator
from dataset.analysis import save_dataset_info


LOGGING_URL = "https://voice-cloning-app-logging.herokuapp.com/"
CONFIG_FILE = "config.ini"


class SocketIOHandler(logging.Handler):
    def emit(self, record):
        text = record.getMessage()
        if text.startswith("Progress"):
            text = text.split("-")[1]
            current, total = text.split("/")
            socketio.emit("progress", {"number": current, "total": total}, namespace="/voice")
        elif text.startswith("Status"):
            socketio.emit("status", {"text": text.replace("Status -", "")}, namespace="/voice")
        else:
            socketio.emit("logs", {"text": text}, namespace="/voice")


# Data
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice")
logger.addHandler(SocketIOHandler())
thread = None


def update_config(data):
    """Writes data to a config file

    Parameters
    ----------
    data : dict
        Dictionary data to write to config file
    """
    config = configparser.ConfigParser()
    config["DEFAULT"] = data

    with open(CONFIG_FILE, "w") as f:
        config.write(f)


def get_config():
    """Gets data from config file

    Returns
    -------
    dict
        Data returned from config file
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config["DEFAULT"]


def can_send_logs():
    """Checks whether logging is allowed.
    Uses config file. If config is not found, defaults to True

    Returns
    -------
    bool
        Whether logging is allowed
    """
    config = get_config()
    if config.get("send_logs") and config["send_logs"] == "False":
        return False
    else:
        return True


def send_error_log(error):
    """Sends error log to server if allowed.

    Parameters
    ----------
    error : dict
        Error object to send (contains type, text & stacktrace)
    """
    if can_send_logs():
        try:
            response = requests.post(LOGGING_URL, data=error)
            if response.status_code != 201:
                print("error logging recieved invalid response")
        except:
            print("error logging failed")


def background_task(func, **kwargs):
    """Runs a background task.
    If function errors out it will send an error log to the error logging server and page.
    Sends 'done' message to frontend when complete.

    Parameters
    ----------
    func : function
        Function to run in background
    kwargs : kwargs
        Kwargs to pass to function
    """
    exception = False
    try:
        socketio.sleep(5)
        func(logging=logger, **kwargs)
    except Exception as e:
        error = {"type": e.__class__.__name__, "text": str(e), "stacktrace": traceback.format_exc()}
        send_error_log(error)
        socketio.emit("error", error, namespace="/voice")
        raise e

    socketio.emit("done", {"text": None}, namespace="/voice")


def start_progress_thread(func, **kwargs):
    """Starts a background task using socketio.

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


def get_next_url(urls, path):
    """Returns the URL of the next step in the voice cloning process.

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
    """Generates a filename suffix using the currrent datetime.

    Returns
    -------
    str
        String suffix
    """
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def delete_folder(path):
    """Deletes a folder.

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
    try:
        with zipfile.ZipFile(dataset, mode="r") as z:
            files_list = z.namelist()
            assert "metadata.csv" in files_list, "Dataset missing metadata.csv. Make sure this file is in the root of the zip file"

            folders = [x.split("/")[0] for x in files_list if "/" in x]
            assert "wavs" in folders, "Dataset missing wavs folder. Make sure this folder is in the root of the zip file"

            wavs = [x for x in files_list if x.startswith("wavs/") and x.endswith(".wav")]
            assert wavs, "No wavs found in wavs folder"

            metadata = z.read("metadata.csv")
            num_metadata_rows = len([row for row in metadata.decode("utf-8").split("\n") if row])
            assert len(wavs) == num_metadata_rows, f"Number of wavs and labels do not match. metadata: {num_metadata_rows}, wavs: {len(wavs)}"

            logging.info("Creating directory")
            os.makedirs(dataset_directory, exist_ok=False)
            os.makedirs(audio_folder, exist_ok=False)

            # Save metadata
            logging.info("Saving files")
            with open(os.path.join(dataset_directory, "metadata.csv"), "wb") as f:
                f.write(metadata)

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
                    clip_lengths.append(librosa.get_duration(filename=new_path))
                    filenames[path] = new_path
                logging.info(f"Progress - {i+1}/{total_wavs}")

            # Get around "file in use" by using delay
            logging.info("Deleting temp files")
            for old_path, new_path in filenames.items():
                os.remove(old_path)
                os.rename(new_path, old_path)

            # Create info file
            logging.info("Creating info file")
            save_dataset_info(
                os.path.join(dataset_directory, "metadata.csv"),
                os.path.join(dataset_directory, "wavs"),
                os.path.join(dataset_directory, "info.json"),
                clip_lengths=clip_lengths
            )
    except Exception as e:
        os.remove(dataset)
        raise e
    
    os.remove(dataset)
