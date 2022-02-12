import os
import io
import zipfile
import traceback
import torch
from datetime import datetime
from pathlib import Path
import re

from main import app, paths
from application.utils import (
    start_progress_thread,
    serve_file,
    get_next_url,
    get_suffix,
    delete_folder,
    import_dataset,
)
from dataset import AUDIO_FOLDER, UNLABELLED_FOLDER, METADATA_FILE, INFO_FILE, CHARACTER_ENCODING
from dataset.create_dataset import create_dataset
from dataset.utils import add_suffix
from dataset.extend_existing_dataset import extend_existing_dataset
from dataset.analysis import get_total_audio_duration, validate_dataset, update_dataset_info
from dataset.transcribe import Silero, DeepSpeech, SILERO_LANGUAGES
from training import TRAIN_FILE, VALIDATION_FILE
from training.train import train, TRAINING_PATH, DEFAULT_ALPHABET
from training.utils import (
    get_available_memory,
    get_gpu_memory,
    get_batch_size,
    load_symbols,
    generate_timelapse_gif,
    create_trainlist_vallist_files,
)
from training.hifigan.train import BATCH_SIZE_PER_GB, CONFIG_FILE
from training.hifigan.train import train as train_hifigan
from training.hifigan.utils import get_checkpoint_options
from synthesis.synthesize import load_model, synthesize
from synthesis.vocoders import Hifigan

from flask import redirect, render_template, request, send_file


URLS = {"/": "Build dataset", "/train": "Train", "/synthesis-setup": "Synthesis"}
ALPHABET_FOLDER = "alphabets"
TEXT_FILE = "text.txt"
SUBTITLE_FILE = "sub.srt"
CHECKPOINT_FOLDER = "checkpoints"
GRAPH_FILE = "graph.png"
RESULTS_FILE = "out.wav"
TEMP_DATASET_UPLOAD = "temp.zip"
TRANSCRIPTION_MODEL = "model.pbmm"
ENGLISH_LANGUAGE = "English"
ALPHABET_FILE = "alphabet.txt"

model = None
vocoder = None
symbols = None


def get_languages():
    silero_languages = {language: True for language in SILERO_LANGUAGES}
    custom_models = {
        language: os.path.isfile(os.path.join(paths["languages"], language, TRANSCRIPTION_MODEL))
        for language in os.listdir(paths["languages"])
    }
    return {**silero_languages, **custom_models}


def get_checkpoints():
    # Checkpoints ordered by name (i.e. checkpoint_0, checkpoint_1000 etc.)
    return {
        model: sorted(
            os.listdir(os.path.join(paths["models"], model)),
            key=lambda name: int(name.split("_")[1]) if "_" in name and name.split("_")[1].isdigit() else 0,
            reverse=True,
        )
        for model in os.listdir(paths["models"])
        if os.listdir(os.path.join(paths["models"], model))
    }


def get_hifigan_checkpoints():
    return {
        model: get_checkpoint_options(os.path.join(paths["hifigan_training"], model))
        for model in os.listdir(paths["hifigan_training"])
        if os.listdir(os.path.join(paths["hifigan_training"], model))
    }


def get_symbols(language):
    if language == ENGLISH_LANGUAGE:
        return DEFAULT_ALPHABET
    elif language in SILERO_LANGUAGES:
        return load_symbols(os.path.join(ALPHABET_FOLDER, f"{language}.txt"))
    else:
        return load_symbols(os.path.join(paths["languages"], language, ALPHABET_FILE))


@app.errorhandler(Exception)
def handle_bad_request(e):
    error = {"type": e.__class__.__name__, "text": str(e), "stacktrace": traceback.format_exc()}
    return render_template("error.html", error=error)


@app.context_processor
def inject_data():
    return {"urls": URLS, "path": request.path}


# Dataset
@app.route("/", methods=["GET"])
def get_create_dataset():
    return render_template("index.html", datasets=os.listdir(paths["datasets"]), languages=get_languages())


@app.route("/", methods=["POST"])
def create_dataset_post():
    min_confidence = float(request.form["confidence"])
    language = request.form["language"]
    combine_clips = request.form.get("combine_clips") is not None
    min_length = float(request.form["min_length"])
    max_length = float(request.form["max_length"])
    transcription_model = (
        Silero(language)
        if language in SILERO_LANGUAGES
        else DeepSpeech(os.path.join(paths["languages"], language, TRANSCRIPTION_MODEL))
    )
    symbols = get_symbols(language)
    text_file = SUBTITLE_FILE if request.files["text_file"].filename.endswith(".srt") else TEXT_FILE

    if request.form["name"]:
        output_folder = os.path.join(paths["datasets"], request.form["name"])
        if os.path.exists(output_folder):
            request.files = None
            raise Exception("Dataset name taken")

        os.makedirs(output_folder, exist_ok=True)
        text_path = os.path.join(output_folder, text_file)
        audio_path = os.path.join(output_folder, request.files["audio_file"].filename)

        with open(text_path, "w", encoding=CHARACTER_ENCODING) as f:
            f.write(request.files["text_file"].read().decode(CHARACTER_ENCODING, "ignore").replace("\r\n", "\n"))
        request.files["audio_file"].save(audio_path)
        start_progress_thread(
            create_dataset,
            text_path=text_path,
            audio_path=audio_path,
            transcription_model=transcription_model,
            output_folder=output_folder,
            min_length=min_length,
            max_length=max_length,
            min_confidence=min_confidence,
            combine_clips=combine_clips,
            symbols=symbols,
        )
    else:
        output_folder = os.path.join(paths["datasets"], request.form["dataset"])
        suffix = get_suffix()
        text_path = os.path.join(output_folder, add_suffix(text_file, suffix))
        audio_path = os.path.join(output_folder, add_suffix(request.files["audio_file"].filename, suffix))

        with open(text_path, "w", encoding=CHARACTER_ENCODING) as f:
            f.write(request.files["text_file"].read().decode(CHARACTER_ENCODING, "ignore").replace("\r\n", "\n"))
        request.files["audio_file"].save(audio_path)

        start_progress_thread(
            extend_existing_dataset,
            text_path=text_path,
            audio_path=audio_path,
            transcription_model=transcription_model,
            output_folder=output_folder,
            suffix=suffix,
            min_length=min_length,
            max_length=max_length,
            min_confidence=min_confidence,
            combine_clips=combine_clips,
            symbols=symbols,
        )

    return render_template("progress.html", next_url=get_next_url(URLS, request.path))


@app.route("/dataset-duration", methods=["GET"])
def get_dataset_duration():
    dataset = request.values["dataset"]
    dataset_error = validate_dataset(os.path.join(paths["datasets"], dataset))
    if not dataset_error:
        return get_total_audio_duration(os.path.join(paths["datasets"], dataset, INFO_FILE))
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
        "train.html",
        cuda_enabled=cuda_enabled,
        batch_size=batch_size,
        datasets=os.listdir(paths["datasets"]),
        checkpoints=get_checkpoints(),
        languages=get_languages(),
    )


@app.route("/train", methods=["POST"])
def train_post():
    language = request.form["language"]
    symbols = get_symbols(language)
    dataset_name = request.form["dataset"]
    epochs = request.form["epochs"]
    batch_size = request.form["batch_size"]
    early_stopping = request.form.get("early_stopping") is not None
    iters_per_checkpoint = request.form["checkpoint_frequency"]
    iters_per_backup_checkpoint = request.form["backup_checkpoint_frequency"]
    train_size = 1 - float(request.form["validation_size"])
    alignment_sentence = request.form["alignment_sentence"]
    multi_gpu = request.form.get("multi_gpu") is not None
    checkpoint_path = (
        os.path.join(paths["models"], dataset_name, request.form["checkpoint"])
        if request.form.get("checkpoint")
        else None
    )

    metadata_path = os.path.join(paths["datasets"], dataset_name, METADATA_FILE)
    use_metadata = os.path.isfile(metadata_path)
    trainlist_path = os.path.join(paths["datasets"], dataset_name, TRAIN_FILE)
    vallist_path = os.path.join(paths["datasets"], dataset_name, VALIDATION_FILE)
    audio_folder = os.path.join(paths["datasets"], dataset_name, AUDIO_FOLDER)
    checkpoint_folder = os.path.join(paths["models"], dataset_name)

    if request.files.get("pretrained_model"):
        transfer_learning_path = os.path.join("data", "pretrained.pt")
        request.files["pretrained_model"].save(transfer_learning_path)
    else:
        transfer_learning_path = None

    start_progress_thread(
        train,
        metadata_path=metadata_path if use_metadata else None,
        trainlist_path=trainlist_path if not use_metadata else None,
        vallist_path=vallist_path if not use_metadata else None,
        audio_directory=audio_folder,
        output_directory=checkpoint_folder,
        symbols=symbols,
        checkpoint_path=checkpoint_path,
        transfer_learning_path=transfer_learning_path,
        epochs=int(epochs),
        batch_size=int(batch_size),
        early_stopping=early_stopping,
        multi_gpu=multi_gpu,
        iters_per_checkpoint=int(iters_per_checkpoint),
        iters_per_backup_checkpoint=int(iters_per_backup_checkpoint),
        train_size=train_size,
        alignment_sentence=alignment_sentence,
    )

    return render_template(
        "progress.html", next_url=get_next_url(URLS, request.path), voice=Path(checkpoint_folder).stem
    )


@app.route("/alignment-timelapse", methods=["GET"])
def download_alignment_timelapse():
    name = request.args.get("name")
    folder = os.path.join(TRAINING_PATH, name)
    output = os.path.join(TRAINING_PATH, f"{name}-training.gif")
    generate_timelapse_gif(folder, output)
    return serve_file(output, f"{name}-training.gif", "image/png", as_attachment=False)


# Synthesis
@app.route("/synthesis-setup", methods=["GET"])
def get_synthesis_setup():
    return render_template(
        "synthesis-setup.html",
        hifigan_models=os.listdir(paths["hifigan"]),
        hifigan_custom_models=get_hifigan_checkpoints(),
        models=os.listdir(paths["models"]),
        checkpoints=get_checkpoints(),
        languages=get_languages(),
    )


@app.route("/synthesis-setup", methods=["POST"])
def synthesis_setup_post():
    global model, vocoder, symbols
    dataset_name = request.form["model"]
    language = request.form["language"]
    symbols = get_symbols(language)
    checkpoint_folder = os.path.join(paths["models"], dataset_name)
    checkpoint = os.path.join(checkpoint_folder, request.form["checkpoint"])
    model = load_model(checkpoint)

    if request.form["vocoder"].startswith("custom-"):
        checkpoint_iteration = request.form["vocoder"].split("-")[1]
        model_path = os.path.join(paths["hifigan_training"], dataset_name, f"g_{checkpoint_iteration}")
        model_config_path = CONFIG_FILE
    else:
        hifigan_folder = os.path.join(paths["hifigan"], request.form["vocoder"])
        model_path = os.path.join(hifigan_folder, "model.pt")
        model_config_path = os.path.join(hifigan_folder, "config.json")

    vocoder = Hifigan(model_path, model_config_path)
    return redirect("/synthesis")


@app.route("/data/<path:path>")
def get_file(path):
    subfolder = path.split("/")[0]
    subpath = path.split("/")[1:-1]
    filename = path.split("/")[-1]
    mimetype = "image/png" if filename.endswith("png") else "audio/wav"
    filepath = os.path.join(paths[subfolder], *subpath, filename)
    return serve_file(filepath, filename, mimetype)


@app.route("/synthesis", methods=["GET", "POST"])
def synthesis_post():
    global model, vocoder, symbols
    if not model or not vocoder or not symbols:
        return redirect("/synthesis-setup")

    if request.method == "GET":
        return render_template("synthesis.html")
    else:
        text = request.form.getlist("text")
        if len(text) == 1:
            text = text[0]
        method = request.form["text_method"]
        split_text = method == "paragraph"
        parent_folder = os.path.join(paths["results"], datetime.now().strftime("%Y-%m"))
        os.makedirs(parent_folder, exist_ok=True)
        first_line = text[0] if type(text) == list else text
        results_folder = os.path.join(
            parent_folder, get_suffix() + "-" + re.sub("[^0-9a-zA-Z _]+", "", first_line.replace(" ", "_"))[:20]
        )
        os.makedirs(results_folder)
        graph_path = os.path.join(results_folder, GRAPH_FILE)
        audio_path = os.path.join(results_folder, RESULTS_FILE)
        graph_web_path = os.path.relpath(graph_path).replace("\\", "/")
        audio_web_path = os.path.relpath(audio_path).replace("\\", "/")
        silence = float(request.form["silence"])
        max_decoder_steps = int(request.form["max_decoder_steps"])

        synthesize(
            model,
            text,
            symbols,
            graph_path,
            audio_path,
            vocoder,
            silence,
            max_decoder_steps=max_decoder_steps,
            split_text=split_text,
        )
        return render_template(
            "synthesis.html",
            text=text,
            method=method,
            graph=graph_web_path,
            audio=audio_web_path,
            silence=silence,
            max_decoder_steps=max_decoder_steps,
        )


# Train hifigan
@app.route("/train-hifigan", methods=["GET"])
def get_train_hifigan():
    cuda_enabled = torch.cuda.is_available()

    if cuda_enabled:
        available_memory = get_gpu_memory(0)
        batch_size = int(available_memory * BATCH_SIZE_PER_GB)
    else:
        batch_size = None

    return render_template(
        "train-hifigan.html",
        cuda_enabled=cuda_enabled,
        batch_size=batch_size,
        datasets=os.listdir(paths["datasets"]),
        checkpoints=get_hifigan_checkpoints(),
    )


# Train hifigan
@app.route("/train-hifigan", methods=["POST"])
def train_hifigan_post():
    dataset_name = request.form["dataset"]
    epochs = request.form["epochs"]
    batch_size = request.form["batch_size"]
    iters_per_checkpoint = request.form["checkpoint_frequency"]
    iters_per_backup_checkpoint = request.form["backup_checkpoint_frequency"]
    train_size = 1 - float(request.form["validation_size"])

    audio_folder = os.path.join(paths["datasets"], dataset_name, AUDIO_FOLDER)
    output_directory = os.path.join(paths["hifigan_training"], dataset_name)

    if request.form.get("checkpoint_iteration"):
        checkpoint_g = os.path.join(
            paths["hifigan_training"], dataset_name, f"g_{request.form['checkpoint_iteration']}"
        )
        checkpoint_do = os.path.join(
            paths["hifigan_training"], dataset_name, f"do_{request.form['checkpoint_iteration']}"
        )
    elif request.files.get("pretrained_model_g"):
        checkpoint_g = os.path.join("data", "pretrained_model_g.pt")
        checkpoint_do = os.path.join("data", "pretrained_model_do.pt")
        request.files["pretrained_model_g"].save(checkpoint_g)
        request.files["pretrained_model_do"].save(checkpoint_do)
    else:
        checkpoint_g = None
        checkpoint_do = None

    start_progress_thread(
        train_hifigan,
        audio_folder=audio_folder,
        output_directory=output_directory,
        checkpoint_g=checkpoint_g,
        checkpoint_do=checkpoint_do,
        epochs=int(epochs),
        batch_size=int(batch_size),
        iters_per_checkpoint=int(iters_per_checkpoint),
        iters_per_backup_checkpoint=int(iters_per_backup_checkpoint),
        train_size=train_size,
    )

    return render_template("progress.html", next_url="/synthesis")


# Manage datasets
@app.route("/manage-datasets", methods=["GET"])
def manage_datasets():
    return render_template(
        "manage-datasets.html", datasets=os.listdir(paths["datasets"]), selected=request.values.get("dataset")
    )


@app.route("/unlabelled-clips", methods=["GET"])
def unlabelled_clips():
    dataset = request.values["dataset"]
    unlabelled_folder = os.path.join(paths["datasets"], dataset, UNLABELLED_FOLDER)
    unlabelled_clips = os.listdir(unlabelled_folder) if os.path.isdir(unlabelled_folder) else []
    return {"unlabelled": unlabelled_clips}


@app.route("/label-clip", methods=["POST"])
def label_clip():
    dataset = request.values["dataset"]
    clip = request.values["unlabelled_clip"]
    text = request.values["sentence"]

    # Update dataset size
    update_dataset_info(
        os.path.join(paths["datasets"], dataset, METADATA_FILE),
        os.path.join(paths["datasets"], dataset, INFO_FILE),
        os.path.join(paths["datasets"], dataset, UNLABELLED_FOLDER, clip),
        text,
    )

    # Add to metadata
    with open(os.path.join(paths["datasets"], dataset, METADATA_FILE), "a") as f:
        f.write(f"{clip}|{text}\n")

    # Move clip
    os.rename(
        os.path.join(paths["datasets"], dataset, UNLABELLED_FOLDER, clip),
        os.path.join(paths["datasets"], dataset, AUDIO_FOLDER, clip),
    )

    return redirect(f"/manage-datasets?dataset={dataset}")


# Import-export
@app.route("/import-export", methods=["GET"])
def import_export():
    return render_template(
        "import-export.html",
        datasets=os.listdir(paths["datasets"]),
        models=os.listdir(paths["models"]),
        checkpoints=get_checkpoints(),
    )


@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    dataset = request.files["dataset"]
    dataset.save(TEMP_DATASET_UPLOAD)
    dataset_name = request.values["name"]
    dataset_directory = os.path.join(paths["datasets"], dataset_name)
    audio_folder = os.path.join(dataset_directory, AUDIO_FOLDER)
    assert not os.path.isdir(dataset_directory), "Output folder already exists"

    start_progress_thread(
        import_dataset, dataset=TEMP_DATASET_UPLOAD, dataset_directory=dataset_directory, audio_folder=audio_folder
    )

    return render_template("progress.html", next_url="/import-export")


@app.route("/download-dataset", methods=["POST"])
def download_dataset():
    dataset_name = request.values["dataset"]
    dataset_directory = os.path.join(paths["datasets"], dataset_name)
    data = io.BytesIO()
    create_trainlist_vallist_files(dataset_directory, os.path.join(dataset_directory, METADATA_FILE))

    with zipfile.ZipFile(data, mode="w") as z:
        z.write(os.path.join(dataset_directory, METADATA_FILE), METADATA_FILE)
        if os.path.isfile(os.path.join(dataset_directory, INFO_FILE)):
            z.write(os.path.join(dataset_directory, INFO_FILE), INFO_FILE)

        audio_directory = os.path.join(dataset_directory, AUDIO_FOLDER)
        for audiofile in os.listdir(audio_directory):
            z.write(os.path.join(audio_directory, audiofile), os.path.join(AUDIO_FOLDER, audiofile))

        z.write(os.path.join(dataset_directory, TRAIN_FILE), TRAIN_FILE)
        z.write(os.path.join(dataset_directory, VALIDATION_FILE), VALIDATION_FILE)

    data.seek(0)

    return send_file(
        data,
        mimetype="application/zip",
        as_attachment=True,
        attachment_filename=f'{dataset_name.replace(" ", "_")}.zip',
    )


@app.route("/upload-model", methods=["POST"])
def upload_model():
    model_name = request.values["name"]
    model_directory = os.path.join(paths["models"], model_name)
    os.makedirs(model_directory, exist_ok=False)

    model_path = os.path.join(model_directory, "model.pt")
    request.files["model_upload"].save(model_path)

    return render_template("import-export.html", message=f"Successfully uploaded {model_name} model")


@app.route("/download-model", methods=["POST"])
def download_model():
    model_name = request.values["model"]
    model_path = os.path.join(paths["models"], model_name, request.values["checkpoint"])

    return send_file(model_path, as_attachment=True, attachment_filename=request.values["checkpoint"])


# Settings
@app.route("/settings", methods=["GET"])
def get_settings():
    return render_template(
        "settings.html",
        datasets=os.listdir(paths["datasets"]),
        models=os.listdir(paths["models"]),
    )


@app.route("/delete-dataset", methods=["POST"])
def delete_dataset_post():
    delete_folder(os.path.join(paths["datasets"], request.values["dataset"]))
    return redirect("/settings")


@app.route("/delete-model", methods=["POST"])
def delete_model_post():
    delete_folder(os.path.join(paths["models"], request.values["model"]))
    return redirect("/settings")


@app.route("/upload-language", methods=["POST"])
def upload_language():
    language = request.values["name"]
    language_dir = os.path.join(paths["languages"], language)
    os.makedirs(language_dir, exist_ok=True)
    if request.files["model"]:
        request.files["model"].save(os.path.join(language_dir, TRANSCRIPTION_MODEL))
    request.files["alphabet"].save(os.path.join(language_dir, ALPHABET_FILE))
    return redirect("/settings")


@app.route("/add-vocoder", methods=["POST"])
def add_vocoder():
    name = request.values["name"]
    hifigan_folder = os.path.join(paths["hifigan"], name)
    os.makedirs(hifigan_folder)
    model_path = os.path.join(hifigan_folder, "model.pt")
    model_config_path = os.path.join(hifigan_folder, "config.json")
    request.files["hifigan-model"].save(model_path)
    request.files["hifigan-config"].save(model_config_path)
    return redirect("/settings")
