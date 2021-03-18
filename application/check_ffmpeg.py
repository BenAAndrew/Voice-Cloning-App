import os
from sys import platform
from subprocess import check_output
import requests
from zipfile import ZipFile


FFMPEG_COMMAND = "ffmpeg -version"
FFMPEG_WINDOWS_INSTALL_PATH = os.path.abspath("ffmpeg\\bin")
FFMPEG_WINDOWS_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_LINUX_INSTALL = "sudo apt-get install -y ffmpeg"


def is_ffmpeg_installed():
    try:
        check_output(FFMPEG_COMMAND.split(" "))
        return True
    except:
        return False


def install_ffmpeg_windows():
    r = requests.get(FFMPEG_WINDOWS_URL)
    with open("ffmpeg.zip", "wb") as f:
        f.write(r.content)

    with ZipFile("ffmpeg.zip", "r") as zipf:
        zipf.extractall()

    ffmpeg_folder = [f for f in os.listdir() if f.startswith("ffmpeg-")][0]
    os.rename(ffmpeg_folder, "ffmpeg")
    os.remove("ffmpeg.zip")


def install_ffmpeg_linux():
    check_output(FFMPEG_LINUX_INSTALL.split(" "))


def check_ffmpeg():
    if not is_ffmpeg_installed():
        if platform == "win32":
            os.environ["PATH"] += os.pathsep + FFMPEG_WINDOWS_INSTALL_PATH
            if not is_ffmpeg_installed():
                print("INSTALLING FFMPEG")
                install_ffmpeg_windows()
                print("VERIFYING FFMPEG INSTALL")
                assert is_ffmpeg_installed(), "FFMPEG did not install correctly"
        else:
            print("INSTALLING FFMPEG")
            install_ffmpeg_linux()
            print("VERIFYING FFMPEG INSTALL")
            assert is_ffmpeg_installed(), "FFMPEG did not install correctly"
