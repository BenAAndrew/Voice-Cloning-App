import os
import sys
import shutil
import webbrowser
from engineio.async_drivers import threading  # noqa
from flask_socketio import SocketIO
from flask import Flask

from application.check_ffmpeg import check_ffmpeg


FOLDERS = ["datasets", "models", "hifigan", "results", "languages", "training", "hifigan_training"]


def get_app_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)


def load_paths():
    base_dir = get_app_path()
    data_folder = os.path.join(base_dir, "data")
    print("Loading data from", data_folder)
    paths = {folder: os.path.join(data_folder, folder) for folder in FOLDERS}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def cleanup_mei():
    """
    Rudimentary workaround for https://github.com/pyinstaller/pyinstaller/issues/2379
    """
    mei_bundle = getattr(sys, "_MEIPASS", None)

    if mei_bundle:
        dir_mei, current_mei = mei_bundle.split("_MEI")
        for file in os.listdir(dir_mei):
            if file.startswith("_MEI") and not file.endswith(current_mei):
                try:
                    shutil.rmtree(os.path.join(dir_mei, file))
                except PermissionError:  # mainly to allow simultaneous pyinstaller instances
                    pass


paths = load_paths()
static = os.path.join("application", "static")
app = Flask(__name__, template_folder=static, static_folder=static)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
socketio = SocketIO(app, async_mode="threading", logger=True, engineio_logger=True, debug=True)
from application.views import *  # noqa

if __name__ == "__main__":
    cleanup_mei()
    check_ffmpeg()
    webbrowser.open_new_tab("http://localhost:5000")
    socketio.run(app, host="0.0.0.0")
