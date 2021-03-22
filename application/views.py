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
    get_prefix,
    send_error_log,
)
from dataset.analysis import get_total_audio_duration
from training.train import train
from training.checkpoint import get_latest_checkpoint
from synthesis.synthesize import load_model, load_waveglow, synthesize
from synthesis.synonyms import get_alternative_word_suggestions

from flask import Flask, request, render_template, redirect, url_for, send_file, send_file


URLS = {"/": "Build dataset", "/train": "Train", "/synthesis-setup": "Synthesis"}
TEXT_FILE = "text.txt"
AUDIO_FILE = "audio.mp3"
ALIGNMENT_FILE = "align.json"
AUDIO_FOLDER = "wavs"
METADATA_FILE = "metadata.csv"
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
    return {
        "urls": URLS,
        "path": request.path,
        "datasets": os.listdir(paths["datasets"]),
        "waveglow_models": os.listdir(paths["waveglow"]),
        "models": os.listdir(paths["models"]),
        "cuda_enabled": torch.cuda.is_available(),
    }


@app.route("/", methods=["GET"], defaults={"endpoint": "index"})
@app.route("/<endpoint>", methods=["GET"])
def get_page(endpoint):
    # Redirect if trying to synthesize before selecting a model
    global model
    if endpoint == "synthesis" and not model:
        return redirect("/synthesis-setup")

    return render_template(f"{endpoint}.html")


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

        request.files["text_file"].save(text_path)
        request.files["audio_file"].save(audio_path)

        start_progress_thread(
            create_dataset,
            text_path=text_path,
            audio_path=audio_path,
            forced_alignment_path=forced_alignment_path,
            output_path=output_path,
            label_path=label_path,
        )
    else:
        output_folder = os.path.join(paths["datasets"], request.form["path"])
        prefix = get_prefix()
        text_path = os.path.join(output_folder, f"text_{prefix}.txt")
        audio_path = os.path.join(output_folder, f"audio_{prefix}.mp3")
        forced_alignment_path = os.path.join(output_folder, f"align_{prefix}.json")

        request.files["text_file"].save(text_path)
        request.files["audio_file"].save(audio_path)

        existing_output_path = os.path.join(output_folder, AUDIO_FOLDER)
        existing_label_path = os.path.join(output_folder, METADATA_FILE)

        start_progress_thread(
            extend_existing_dataset,
            text_path=text_path,
            audio_path=audio_path,
            forced_alignment_path=forced_alignment_path,
            output_path=existing_output_path,
            label_path=existing_label_path,
            prefix=prefix,
        )

    return render_template("progress.html", next_url=get_next_url(URLS, request.path))


@app.route("/dataset-duration", methods=["GET"])
def get_dataset_duration():
    dataset = request.values["dataset"]
    duration = get_total_audio_duration(os.path.join(paths["datasets"], dataset, "metadata.csv"))
    return str(duration)


@app.route("/train", methods=["POST"])
def train_post():
    dataset_name = request.form["path"]
    epochs = request.form["epochs"]

    metadata_path = os.path.join(paths["datasets"], dataset_name, METADATA_FILE)
    audio_folder = os.path.join(paths["datasets"], dataset_name, AUDIO_FOLDER)
    checkpoint_folder = os.path.join(paths["models"], dataset_name)

    if request.files.get("pretrained_model"):
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_folder, "pretrained.pt")
        request.files["pretrained_model"].save(checkpoint_path)
    else:
        checkpoint_path = None

    start_progress_thread(
        train,
        metadata_path=metadata_path,
        dataset_directory=audio_folder,
        output_directory=checkpoint_folder,
        checkpoint_path=checkpoint_path,
        epochs=int(epochs),
    )

    return render_template("progress.html", next_url=get_next_url(URLS, request.path))


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


@app.route("/synthesis", methods=["POST"])
def synthesis_post():
    global model, waveglow_model

    text = request.form["text"]
    folder_name = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    folder_name = folder_name.replace(" ", "_")
    folder_name += f"_{get_prefix()}"
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


@app.route("/import-export", methods=["GET"])
def import_export():
    return render_template("import-export.html")


@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    dataset = request.files["dataset"]
    dataset_name = request.values["name"]
    dataset_directory = os.path.join(paths["datasets"], dataset_name)
    audio_folder = os.path.join(dataset_directory, AUDIO_FOLDER)
    os.makedirs(dataset_directory, exist_ok=False)
    os.makedirs(audio_folder, exist_ok=False)

    with zipfile.ZipFile(dataset, mode="r") as z:
        files_list = z.namelist()
        if "metadata.csv" not in files_list:
            render_template("import-export.html", message=f"Dataset missing metadata.csv")
        if "wavs" not in files_list:
            render_template("import-export.html", message=f"Dataset missing wavs folder")

        # Save metadata
        with open(os.path.join(dataset_directory, "metadata.csv"), "wb") as f:
            data = z.read("metadata.csv")
            f.write(data)

        # Save wavs
        for name in files_list:
            if name.endswith(".wav"):
                data = z.read(name)
                path = os.path.join(dataset_directory, "wavs", name.split("/")[1])
                with open(path, "wb") as f:
                    f.write(data)

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
