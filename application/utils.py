import logging
from threading import Thread
import os
from datetime import datetime
import requests
import traceback
import configparser
import shutil
import subprocess
import time

from main import socketio
from dataset.audio_processing import convert_audio
from dataset.clip_generator import clip_generator, extend_dataset
from dataset.analysis import save_dataset_info


LOGGING_URL = "https://voice-cloning-app-logging.herokuapp.com/"
LOG_FILENAME = "app.log"
MAX_PROCESS_START_WAIT = 30


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


def create_dataset(text_path, audio_path, forced_alignment_path, output_path, label_path, info_path, logging):
    logging.info(f"Coverting {audio_path}...")
    converted_audio = convert_audio(audio_path)
    clip_generator(converted_audio, text_path, forced_alignment_path, output_path, label_path, logging=logging)
    logging.info("Getting dataset info...")
    save_dataset_info(label_path, output_path, info_path)


def extend_existing_dataset(
    text_path, audio_path, forced_alignment_path, output_path, label_path, suffix, info_path, logging
):
    assert os.path.isdir(output_path), "Missing existing dataset clips folder"
    assert os.path.isfile(label_path), "Missing existing dataset metadata file"
    logging.info(f"Coverting {audio_path}...")
    converted_audio = convert_audio(audio_path)
    extend_dataset(converted_audio, text_path, forced_alignment_path, output_path, label_path, suffix, logging=logging)
    logging.info("Getting dataset info...")
    save_dataset_info(label_path, output_path, info_path)


def update_config(data):
    config = configparser.ConfigParser()
    config["DEFAULT"] = data

    with open("config.ini", "w") as f:
        config.write(f)


def get_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["DEFAULT"]


def can_send_logs():
    config = get_config()
    if config.get("send_logs") and config["send_logs"] == "False":
        return False
    else:
        return True


def send_error_log(error):
    if can_send_logs():
        try:
            response = requests.post(LOGGING_URL, data=error)
            if response.status_code != 201:
                print("error logging recieved invalid response")
        except:
            print("error logging failed")


def handle_process_error(message):
    error = {"type": "", "text": message, "stacktrace": ""}
    socketio.emit("error", error, namespace="/voice")
    send_error_log(error)


def background_task(command):
    # Remove existing log file
    if os.path.isfile(LOG_FILENAME):
        os.remove(LOG_FILENAME)

    command.extend(["-v", LOG_FILENAME])
    print(command)
    # Start subprocess
    popen = subprocess.Popen(command, stderr=subprocess.PIPE)

    # Wait for process to start & create log file
    for _ in range(MAX_PROCESS_START_WAIT * 2):
        if not os.path.isfile(LOG_FILENAME):
            time.sleep(0.5)

    if not os.path.isfile(LOG_FILENAME):
        try:
            _, err = popen.communicate()
            error_message = err.decode("utf-8")
        except:
            error_message = "Process failed to start and create log file"
        handle_process_error(error_message)

    # Read & send logs to the frontend
    with open(LOG_FILENAME, "r", encoding="utf-8") as f:
        line = ""
        while popen.poll() is None:
            line = f.readline().strip()
            if line:
                logger.info(line)

    if popen.returncode != 0:
        # Read process error
        _, err = popen.communicate()
        handle_process_error(err.decode("utf-8"))
    else:
        socketio.emit("done", {"text": None}, namespace="/voice")


def start_progress_thread(command):
    global thread
    print("Starting Thread")
    thread = socketio.start_background_task(background_task, command=command)


def get_next_url(urls, path):
    urls = list(urls.keys())
    next_url_index = urls.index(path) + 1
    return urls[next_url_index] if next_url_index < len(urls) else ""


def get_suffix():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def delete_folder(path):
    assert os.path.isdir(path), f"{path} does not exist"
    shutil.rmtree(path)
