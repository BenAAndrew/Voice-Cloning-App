import logging
import requests
import os
import time
import shutil

from application.utils import handle_logged_message

TEMP_ZIP = "data\\temp"
REQUEST_INTERVAL = 5


def download_checkpoint(remote_url, folder):
    r = requests.get(remote_url+"download")  
    with open(os.path.join(folder, "checkpoint.pt"), "wb") as f:
        f.write(r.content)


def handle_server_message(message, remote_url, output_directory):
    if message.startswith("ERROR -"):
        raise Exception(message.split("-")[1])
    elif message.startswith("Saving"):
        download_checkpoint(remote_url, output_directory)
    elif message == "DONE":
        return True
    else:
        handle_logged_message(message)
    return False


def remote_train(
    remote_url,
    dataset_folder,
    output_directory,
    find_checkpoint=True,
    checkpoint_path=None,
    transfer_learning_path=None,
    overwrite_checkpoints=True,
    epochs=8000,
    batch_size=None,
    early_stopping=True,
    iters_per_checkpoint=1000,
    logging=logging,
):
    assert os.path.isdir(dataset_folder), "Dataset folder not found"
    logging.info("Compressing dataset")
    shutil.make_archive(TEMP_ZIP, "zip", dataset_folder)

    logging.info("Sending data to server")
    files = {"dataset": open(TEMP_ZIP+".zip","rb")}
    if transfer_learning_path:
        files["transfer_learning"] = open(transfer_learning_path,"rb")
    values = {"epochs": epochs, "batch_size": batch_size, "early_stopping": early_stopping, "iters_per_checkpoint": iters_per_checkpoint, "overwrite_checkpoints": overwrite_checkpoints}
    r = requests.post(remote_url, data=values, files=files)
    assert r.status_code == 200, f"Server did not handle request (returned {r.status_code})"
    logging.info("Data sent")

    os.makedirs(output_directory, exist_ok=True)
    log_index = 0
    complete = False 
    while not complete:
        r = requests.get(remote_url)
        if r.text:
            logs = r.text.split("\n")
            if len(logs) > log_index:
                for i in range(log_index, len(logs)):
                    complete = handle_server_message(logs[i], remote_url, output_directory)
                log_index = len(logs)
        time.sleep(REQUEST_INTERVAL)
