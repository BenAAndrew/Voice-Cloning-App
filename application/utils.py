import logging
from threading import Thread
import os
from datetime import datetime
import requests
import traceback

from main import socketio
from dataset.audio_processing import convert_audio
from dataset.clip_generator import clip_generator, extend_dataset


LOGGING_URL = "https://voice-cloning-app-logging.herokuapp.com/"


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


def create_dataset(text_path, audio_path, forced_alignment_path, output_path, label_path, logging):
    logging.info(f"Coverting {audio_path}...")
    converted_audio = convert_audio(audio_path)
    clip_generator(converted_audio, text_path, forced_alignment_path, output_path, label_path, logging=logging)


def extend_existing_dataset(text_path, audio_path, forced_alignment_path, output_path, label_path, suffix, logging):
    assert os.path.isdir(output_path), "Missing existing dataset clips folder"
    assert os.path.isfile(label_path), "Missing existing dataset metadata file"
    logging.info(f"Coverting {audio_path}...")
    converted_audio = convert_audio(audio_path)
    extend_dataset(converted_audio, text_path, forced_alignment_path, output_path, label_path, suffix, logging=logging)


def send_error_log(error):
    try:
        response = requests.post(LOGGING_URL, data=error)
        if response.status_code != 201:
            print("error logging recieved invalid response")
    except:
        print("error logging failed")


def background_task(func, **kwargs):
    exception = False
    try:
        socketio.sleep(5)
        func(logging=logger, **kwargs)
    except Exception as e:
        error = {"type": e.__class__.__name__, "text": str(e), "stacktrace": traceback.format_exc()}
        send_error_log(error)
        socketio.emit("error", error, namespace="/voice")
        exception = True
        print(e)

    if not exception:
        socketio.emit("done", {"text": None}, namespace="/voice")


def start_progress_thread(func, **kwargs):
    global thread
    print("Starting Thread")
    thread = socketio.start_background_task(background_task, func=func, **kwargs)


def get_next_url(urls, path):
    urls = list(urls.keys())
    next_url_index = urls.index(path) + 1
    return urls[next_url_index] if next_url_index < len(urls) else ""


def get_suffix():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
