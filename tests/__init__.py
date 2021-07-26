import os
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_if_not_exists(path, id):
    if not os.path.isfile(path):
        gdd.download_file_from_google_drive(file_id=id, dest_path=path)
        assert os.path.isfile(path)
