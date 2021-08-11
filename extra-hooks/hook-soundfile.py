import os
from PyInstaller.utils.hooks import get_package_paths

sfp = get_package_paths("soundfile")

bins = os.path.join(sfp[0], "_soundfile_data")
datas = [(bins, "_soundfile_data")]
binaries = []
