import os
import shutil
import PyInstaller.__main__
from PyInstaller.utils.hooks import get_package_paths


if __name__ == "__main__":
    assert os.path.isfile("ffmpeg\\bin\\ffmpeg.exe")
    assert os.path.isfile("en_v5.jit")
    pytorch_libs = os.path.join(get_package_paths("torch")[1], "lib")
    PyInstaller.__main__.run(
        [
            "main.py",
            "--onefile",
            "--clean",
            "--icon=application\static\\favicon\\app-icon.ico",
            "--add-data=application/static;application/static",
            "--additional-hooks=extra-hooks",
            "--hidden-import=sklearn.utils._weight_vector",
            "--hidden-import=sklearn.utils._cython_blas",
            "--hidden-import=sklearn.neighbors._typedefs",
            "--hidden-import=sklearn.neighbors._quad_tree",
            "--hidden-import=sklearn.tree._utils",
            "--hidden-import=sklearn.tree",
            "--hidden-import=scipy.special.cython_special",
            "--add-data=ffmpeg/bin/ffmpeg.exe;.",
            "--add-data=en_v5.jit;.",
            "--add-data",
            pytorch_libs + ";.",
        ]
    )
    shutil.rmtree("build")
