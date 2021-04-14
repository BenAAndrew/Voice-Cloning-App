import os
import sys
import re
import inflect
import io
import zipfile
import traceback
import torch

sys.path.append("synthesis/waveglow/")

from main import app, paths
from application.utils import (
    start_progress_thread,
    get_next_url,
    create_dataset,
    extend_existing_dataset,
    get_suffix,
    send_error_log,
    update_config,
    can_send_logs,
    delete_folder,
)
from dataset.analysis import get_total_audio_duration, validate_dataset, save_dataset_info
from training.train import train
from training.checkpoint import get_latest_checkpoint
from training.utils import get_available_memory, get_batch_size
from synthesis.synthesize import load_model, load_waveglow, synthesize
from synthesis.synonyms import get_alternative_word_suggestions

from flask import Flask, request, render_template, redirect, url_for, send_file


URLS = {"/": "Build dataset", "/train": "Train", "/synthesis-setup": "Synthesis"}
TEXT_FILE = "text.txt"
AUDIO_FILE = "audio.mp3"
ALIGNMENT_FILE = "align.json"
AUDIO_FOLDER = "wavs"
METADATA_FILE = "metadata.csv"
INFO_FILE = "info.json"
CHECKPOINT_FOLDER = "checkpoints"
GRAPH_FILE = "graph.png"
RESULTS_FILE = "out.wav"

model = None
waveglow_model = None
inflect_engine = inflect.engine()


@app.errorhandler(Exception)
def handle_bad_request(e):
    error = {"type": e.__class__.__name__, "text": str(e), "stacktrace": traceback.format_exc()}
    send_error_log(error)
    return render_template("error.html", error=error)


@app.context_processor
def inject_data():
    return {"urls": URLS, "path": request.path}


# Dataset
@app.route("/", methods=["GET"])
def get_create_dataset():
    return render_template("index.html", datasets=os.listdir(paths["datasets"]))


@app.route("/datasource", methods=["GET"])
def get_datasource():
    return render_template("datasource.html")


@app.route("/", methods=["POST"])
def create_dataset_post():
    if request.form["name"]:
        output_folder = os.path.join(paths["datasets"], request.form["name"])
        if os.path.exists(output_folder):
            request.files = None
            raise Exception("Dataset name taken")
        os.makedirs(output_folder, exist_ok=True)

        text_path = os.path.join(output_folder, TEXT_FILE)
        audio_path = os.path.join(output_folder, AUDIO_FILE)
        forced_alignment_path = os.path.join(output_folder, ALIGNMENT_FILE)
        output_path = os.path.join(output_folder, AUDIO_FOLDER)
        label_path = os.path.join(output_folder, METADATA_FILE)
        info_path = os.path.join(output_folder, INFO_FILE)

        request.files["text_file"].save(text_path)
        request.files["audio_file"].save(audio_path)

        start_progress_thread(
            [
                "python", "dataset\\generate_dataset.py", 
                "-t", text_path, 
                "-a", audio_path, 
                "-f", forced_alignment_path,
                "-o", output_path,
                "-l", label_path,
                "-i", info_path,
            ]
        )
    else:
        output_folder = os.path.join(paths["datasets"], request.form["path"])
        suffix = get_suffix()
        text_path = os.path.join(output_folder, f"text-{suffix}.txt")
        audio_path = os.path.join(output_folder, f"audio-{suffix}.mp3")
        forced_alignment_path = os.path.join(output_folder, f"align-{suffix}.json")
        info_path = os.path.join(output_folder, INFO_FILE)

        request.files["text_file"].save(text_path)
        request.files["audio_file"].save(audio_path)

        existing_output_path = os.path.join(output_folder, AUDIO_FOLDER)
        existing_label_path = os.path.join(output_folder, METADATA_FILE)

        start_progress_thread(
            [
                "python", "dataset\\extend_existing_dataset.py", 
                "-t", text_path, 
                "-a", audio_path, 
                "-f", forced_alignment_path,
                "-o", existing_output_path,
                "-l", existing_label_path,
                "-s", suffix,
                "-i", info_path,
            ]
        )

    return render_template("progress.html", next_url=get_next_url(URLS, request.path))


@app.route("/dataset-duration", methods=["GET"])
def get_dataset_duration():
    dataset = request.values["dataset"]
    dataset_error = validate_dataset(
        os.path.join(paths["datasets"], dataset), metadata_file=METADATA_FILE, audio_folder=AUDIO_FOLDER
    )
    if not dataset_error:
        duration, total_clips = get_total_audio_duration(os.path.join(paths["datasets"], dataset, INFO_FILE))
        return {"duration": duration, "total_clips": total_clips}
    else:
        return {"error": dataset_error}


# Training
@app.route("/train", methods=["GET"])
def get_train():
    cuda_enabled = torch.cuda.is_available()

    if cuda_enabled:
        available_memory_gb = get_available_memory()
        batch_size = get_batch_size(available_memory_gb)
    else:
        batch_size = None

    return render_template(
        "train.html", cuda_enabled=cuda_enabled, batch_size=batch_size, datasets=os.listdir(paths["datasets"])
    )


@app.route("/train", methods=["POST"])
def train_post():
    dataset_name = request.form["path"]
    epochs = request.form["epochs"]
    batch_size = request.form["batch_size"]

    metadata_path = os.path.join(paths["datasets"], dataset_name, METADATA_FILE)
    audio_folder = os.path.join(paths["datasets"], dataset_name, AUDIO_FOLDER)
    checkpoint_folder = os.path.join(paths["models"], dataset_name)
    pretrained_folder = os.path.join(paths["pretrained"], dataset_name)

    if request.files.get("pretrained_model"):
        os.makedirs(pretrained_folder, exist_ok=True)
        transfer_learning_path = os.path.join(pretrained_folder, "pretrained.pt")
        request.files["pretrained_model"].save(transfer_learning_path)
    else:
        transfer_learning_path = None

    command = [
        "python", "training\\train.py", 
        "-m", metadata_path, 
        "-d", audio_folder, 
        "-o", checkpoint_folder,
        "-e", epochs,
        "-b", batch_size,
    ]

    if transfer_learning_path:
        command.extend(["-t", transfer_learning_path])

    start_progress_thread(command)
    return render_template("progress.html", next_url=get_next_url(URLS, request.path))


# Synthesis
@app.route("/synthesis-setup", methods=["GET"])
def get_synthesis_setup():
    return render_template(
        "synthesis-setup.html", waveglow_models=os.listdir(paths["waveglow"]), models=os.listdir(paths["models"])
    )


@app.route("/synthesis-setup", methods=["POST"])
def synthesis_setup_post():
    global model, waveglow_model

    if request.files.get("waveglow"):
        waveglow_path = os.path.join(paths["waveglow"], request.files["waveglow"].filename)
        request.files["waveglow"].save(waveglow_path)
    elif request.form.get("existing_waveglow"):
        waveglow_path = os.path.join(paths["waveglow"], request.form["existing_waveglow"])
    else:
        return render_template("synthesis-setup.html", path=None, error="No waveglow model chosen")

    dataset_name = request.form["path"]
    checkpoint_folder = os.path.join(paths["models"], dataset_name)
    checkpoint = get_latest_checkpoint(checkpoint_folder)
    model = load_model(checkpoint)
    waveglow_model = load_waveglow(waveglow_path)

    return redirect("/synthesis")


@app.route("/data/results/<path:path>")
def get_result_file(path):
    filename = path.split("/")[-1]
    mimetype = "image/png" if filename.endswith("png") else "audio/wav"

    with open(os.path.join(paths["results"], path), "rb") as f:
        return send_file(io.BytesIO(f.read()), attachment_filename=filename, mimetype=mimetype, as_attachment=True)


@app.route("/synthesis", methods=["GET", "POST"])
def synthesis_post():
    global model, waveglow_model
    if not model:
        return redirect("/synthesis-setup")

    if request.method == "GET":
        return render_template("synthesis.html")
    else:
        text = request.form["text"]
        folder_name = re.sub(r"[^A-Za-z0-9 ]+", "", text)
        folder_name = folder_name.replace(" ", "_")
        folder_name += f"_{get_suffix()}"
        results_folder = os.path.join(paths["results"], folder_name)
        os.makedirs(results_folder)
        graph_path = os.path.join(results_folder, GRAPH_FILE)
        audio_path = os.path.join(results_folder, RESULTS_FILE)
        graph_web_path = graph_path.replace("\\", "/")
        audio_web_path = audio_path.replace("\\", "/")

        synthesize(model, waveglow_model, text, inflect_engine, graph_path, audio_path)
        return render_template(
            "synthesis.html",
            text=text.strip(),
            alertnative_words=get_alternative_word_suggestions(audio_path, text),
            graph=graph_web_path,
            audio=audio_web_path,
        )


# Import-export
@app.route("/import-export", methods=["GET"])
def import_export():
    return render_template("import-export.html")


@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    dataset = request.files["dataset"]
    dataset_name = request.values["name"]
    dataset_directory = os.path.join(paths["datasets"], dataset_name)
    audio_folder = os.path.join(dataset_directory, AUDIO_FOLDER)
    assert not os.path.isdir(dataset_directory), "Output folder already exists"

    with zipfile.ZipFile(dataset, mode="r") as z:
        files_list = z.namelist()
        if "metadata.csv" not in files_list:
            return render_template(
                "import-export.html",
                message="Dataset missing metadata.csv. Make sure this file is in the root of the zip file",
            )

        folders = [x.split("/")[0] for x in files_list if "/" in x]
        if "wavs" not in folders:
            return render_template(
                "import-export.html",
                message="Dataset missing wavs folder. Make sure this folder is in the root of the zip file",
            )

        wavs = [x for x in files_list if x.startswith("wavs/") and x.endswith(".wav")]
        if not wavs:
            return render_template("import-export.html", message="No wavs found in wavs folder")

        os.makedirs(dataset_directory, exist_ok=False)
        os.makedirs(audio_folder, exist_ok=False)

        # Save metadata
        with open(os.path.join(dataset_directory, "metadata.csv"), "wb") as f:
            data = z.read("metadata.csv")
            f.write(data)

        # Save wavs
        for wav in wavs:
            data = z.read(wav)
            path = os.path.join(dataset_directory, "wavs", wav.split("/")[1])
            with open(path, "wb") as f:
                f.write(data)

        # Create info file
        save_dataset_info(
            os.path.join(dataset_directory, "metadata.csv"),
            os.path.join(dataset_directory, "wavs"),
            os.path.join(dataset_directory, "info.json"),
        )

    return render_template("import-export.html", message=f"Successfully uploaded {dataset_name} dataset")


@app.route("/download-dataset", methods=["POST"])
def download_dataset():
    dataset_name = request.values["dataset"]
    dataset_directory = os.path.join(paths["datasets"], dataset_name)
    data = io.BytesIO()

    with zipfile.ZipFile(data, mode="w") as z:
        for filename in os.listdir(dataset_directory):
            if os.path.isfile(os.path.join(dataset_directory, filename)):
                z.write(os.path.join(dataset_directory, filename), filename)

        audio_directory = os.path.join(dataset_directory, AUDIO_FOLDER)
        for audiofile in os.listdir(audio_directory):
            z.write(os.path.join(audio_directory, audiofile), os.path.join(AUDIO_FOLDER, audiofile))

    data.seek(0)

    return send_file(
        data,
        mimetype="application/zip",
        as_attachment=True,
        attachment_filename=f'{dataset_name.replace(" ", "_")}.zip',
    )


@app.route("/upload-model", methods=["POST"])
def upload_model():
    model = request.files["model"]
    model_name = request.values["name"]
    model_directory = os.path.join(paths["models"], model_name)
    os.makedirs(model_directory, exist_ok=False)

    model_path = os.path.join(model_directory, "model.pt")
    request.files["model"].save(model_path)

    return render_template("import-export.html", message=f"Successfully uploaded {model_name} model")


@app.route("/download-model", methods=["POST"])
def download_model():
    model_name = request.values["model"]
    model_path = get_latest_checkpoint(os.path.join(paths["models"], model_name))

    return send_file(model_path, as_attachment=True, attachment_filename=f'{model_name.replace(" ", "_")}.pt')


# Settings
@app.route("/settings", methods=["GET"])
def get_settings():
    print(can_send_logs())
    return render_template(
        "settings.html",
        send_logs=can_send_logs(),
        datasets=os.listdir(paths["datasets"]),
        models=os.listdir(paths["models"]),
    )


@app.route("/update-config", methods=["POST"])
def update_config_post():
    send_logs = request.values["send_logs"]
    update_config({"send_logs": send_logs})
    return redirect("/settings")


@app.route("/delete-dataset", methods=["POST"])
def delete_dataset_post():
    delete_folder(os.path.join(paths["datasets"], request.values["dataset"]))
    return redirect("/settings")


@app.route("/delete-model", methods=["POST"])
def delete_model_post():
    delete_folder(os.path.join(paths["models"], request.values["model"]))
    return redirect("/settings")
