import os
import sys
from sys import platform
from subprocess import check_output
import requests
from zipfile import ZipFile


FFMPEG_COMMAND = "ffmpeg -version"
FFMPEG_PATHS = [os.path.abspath(os.path.join(getattr(sys, "_MEIPASS", ""))), os.path.abspath("ffmpeg\\bin")]
FFMPEG_WINDOWS_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_LINUX_INSTALL = "sudo apt-get install -y ffmpeg"


def is_ffmpeg_installed():
    """Checks if FFmpeg is installed

    Returns
    -------
    bool
        Whether or not ffmpeg is installed
    """
    try:
        check_output(FFMPEG_COMMAND.split(" "))
        return True
    except:
        return False


def install_ffmpeg_windows():
    """Downloads and extracts the FFmpeg library"""
    try:
        r = requests.get(FFMPEG_WINDOWS_URL)
        with open("ffmpeg.zip", "wb") as f:
            f.write(r.content)

        with ZipFile("ffmpeg.zip", "r") as zipf:
            zipf.extractall()

        ffmpeg_folder = [f for f in os.listdir() if f.startswith("ffmpeg-")][0]
        os.rename(ffmpeg_folder, "ffmpeg")
        os.remove("ffmpeg.zip")
        os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg\\bin")
    except requests.exceptions.ConnectionError:
        raise Exception("Unable to download FFmpeg. Please install manually")


def install_ffmpeg_linux():
    """Runs the linux FFmpeg install command"""
    check_output(FFMPEG_LINUX_INSTALL.split(" "))


def try_ffmpeg_paths():
    """
    Try ffmpeg paths to find existing install

    Returns
    -------
    str
        Path to existing install (or None if not found)
    """
    for path in FFMPEG_PATHS:
        if os.path.isdir(path):
            os.environ["PATH"] += os.pathsep + path
            if is_ffmpeg_installed():
                return path
    return None


def check_ffmpeg():
    """Checks if FFmpeg is installed, and if not will install

    Raises
    -------
    AssertionError
        If ffmpeg could not be installed
    """
    if not is_ffmpeg_installed():
        if platform == "win32":
            existing_install = try_ffmpeg_paths()
            if existing_install:
                print("USING FFMPEG INSTALL", existing_install)
            else:
                print("INSTALLING FFMPEG")
                install_ffmpeg_windows()
                print("VERIFYING FFMPEG INSTALL")
                assert is_ffmpeg_installed(), "FFMPEG did not install correctly"
        else:
            print("INSTALLING FFMPEG")
            install_ffmpeg_linux()
            print("VERIFYING FFMPEG INSTALL")
            assert is_ffmpeg_installed(), "FFMPEG did not install correctly"
