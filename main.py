import os
import sys
import shutil
import webbrowser
from engineio.async_drivers import threading  # noqa
from flask_socketio import SocketIO
from flask import Flask

from application.check_ffmpeg import check_ffmpeg


def load_paths():
    paths = {
        "datasets": os.path.join("data", "datasets"),
        "models": os.path.join("data", "models"),
        "pretrained": os.path.join("data", "pretrained"),
        "hifigan": os.path.join("data", "hifigan"),
        "results": os.path.join("data", "results"),
        "languages": os.path.join("data", "languages"),
        "training": os.path.join("data", "training"),
    }
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
    socketio.run(app)
