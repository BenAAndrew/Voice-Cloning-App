import os


if __name__ == "__main__":
    import PyInstaller.__main__
    from PyInstaller.utils.hooks import collect_data_files

    site_packages_folder = "D:\Program Files\\anaconda3\envs\pyinstaller11\Lib\site-packages"
    deepspeech_dll = os.path.join(site_packages_folder, "deepspeech", "lib", "libdeepspeech.so")
    llvmlite_dll = os.path.join(site_packages_folder, "llvmlite", "binding", "llvmlite.dll")
    kaiser_filters = os.path.join(site_packages_folder, "resampy", "data")
    pytorch_libs = os.path.join(site_packages_folder, "torch", "lib")

    PyInstaller.__main__.run([
        "main.py",
        "--onefile",
        "--clean",
        "--hidden-import=sklearn.utils._weight_vector",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors._typedefs",
        "--hidden-import=sklearn.neighbors._quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=sklearn.tree",
        "--hidden-import=scipy.special.cython_special",
        "--icon=application\static\\favicon\\app-icon.ico", 
        "--add-data=application/static;application/static",
        "--add-data",
        deepspeech_dll+";.",
        "--add-data",
        llvmlite_dll+";.",
        "--paths",
        pytorch_libs,
        "--additional-hooks=extra-hooks",
    ])
