import os
import logging
import webbrowser
from engineio.async_drivers import threading
from flask_socketio import SocketIO
from flask import Flask

from application.check_ffmpeg import check_ffmpeg


def load_paths():
    paths = {
        "datasets": os.path.join("data", "datasets"),
        "models": os.path.join("data", "models"),
        "pretrained": os.path.join("data", "pretrained"),
        "waveglow": os.path.join("data", "waveglow"),
        "hifigan": os.path.join("data", "hifigan"),
        "results": os.path.join("data", "results"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


paths = load_paths()
static = os.path.join("application", "static")
app = Flask(__name__, template_folder=static, static_folder=static)
socketio = SocketIO(app, async_mode="threading", logger=True, engineio_logger=True, debug=True)
from application.views import *

if __name__ == "__main__":
    check_ffmpeg()
    webbrowser.open_new_tab("http://localhost:5000")
    socketio.run(app)
