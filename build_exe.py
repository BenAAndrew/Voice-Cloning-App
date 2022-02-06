import os
import shutil
import PyInstaller.__main__
from PyInstaller.utils.hooks import get_package_paths


if __name__ == "__main__":
    assert os.path.isfile("ffmpeg\\bin\\ffmpeg.exe")
    assert os.path.isfile("en_v5.jit")
    assert os.path.isfile("application\\static\\favicon\\app-icon.ico")
    pytorch_libs = os.path.join(get_package_paths("torch")[1], "lib")
    PyInstaller.__main__.run(
        [
            "main.py",
            "--onefile",
            "--clean",
            "--icon=application\\static\\favicon\\app-icon.ico",
            "--additional-hooks=extra-hooks",
            "--hidden-import=scipy.special.cython_special",
            "--add-data=application/static;application/static",
            "--add-data=alphabets;alphabets",
            "--add-data=training/hifigan;training/hifigan",
            "--add-data=ffmpeg/bin/ffmpeg.exe;.",
            "--add-data=en_v5.jit;.",
        ]
    )
    shutil.rmtree("build")
