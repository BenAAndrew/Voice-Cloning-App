import os
import io
import zipfile
import traceback
import torch
from datetime import datetime
from pathlib import Path
import re

from main import app, paths
from application import constants
from application.utils import (
    get_symbols,
    start_progress_thread,
    serve_file,
    get_next_url,
    get_suffix,
    delete_folder,
    import_dataset,
)
from dataset.create_dataset import (
    create_dataset,
    AUDIO_FOLDER,
    UNLABELLED_FOLDER,
    METADATA_FILE,
    ALIGNMENT_FILE,
    INFO_FILE,
)
from dataset.clip_generator import add_suffix
from dataset.extend_existing_dataset import extend_existing_dataset
from dataset.analysis import get_total_audio_duration, validate_dataset
from dataset.transcribe import Silero, DeepSpeech
from training.train import train, TRAINING_PATH
from training.utils import get_available_memory, get_batch_size, generate_timelapse_gif
from synthesis.synthesize import load_model, synthesize
from synthesis.vocoders import Hifigan

from flask import redirect, render_template, request, send_file




model = None
vocoder = None
symbols = None


def get_languages():
    silero_languages = {language: True for language in constants.SILERO_LANGUAGES}
    custom_models = {
        language: os.path.isfile(os.path.join(paths["languages"], language, constants.TRANSCRIPTION_MODEL))
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



@app.errorhandler(Exception)
def handle_bad_request(e):
    error = {"type": e.__class__.__name__, "text": str(e), "stacktrace": traceback.format_exc()}
    return render_template("error.html", error=error)


@app.context_processor
def inject_data():
    return {"urls": constants.URLS, "path": request.path}


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
        if language in constants.SILERO_LANGUAGES
        else DeepSpeech(language)
    )
    text_file = constants.SUBTITLE_FILE if request.files["text_file"].filename.endswith(".srt") else constants.TEXT_FILE

    if request.form["name"]:
        output_folder = os.path.join(paths["datasets"], request.form["name"])
        if os.path.exists(output_folder):
            request.files = None
            raise Exception("Dataset name taken")

        os.makedirs(output_folder, exist_ok=True)
        text_path = os.path.join(output_folder, text_file)
        audio_path = os.path.join(output_folder, request.files["audio_file"].filename)

        with open(text_path, "w", encoding=constants.CHARACTER_ENCODING) as f:
            f.write(request.files["text_file"].read().decode(constants.CHARACTER_ENCODING, "ignore").replace("\r\n", "\n"))
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
        )
    else:
        output_folder = os.path.join(paths["datasets"], request.form["dataset"])
        suffix = get_suffix()
        text_path = os.path.join(output_folder, add_suffix(text_file, suffix))
        audio_path = os.path.join(output_folder, add_suffix(request.files["audio_file"].filename, suffix))
        forced_alignment_path = os.path.join(output_folder, add_suffix(ALIGNMENT_FILE, suffix))
        info_path = os.path.join(output_folder, add_suffix(INFO_FILE, suffix))

        with open(text_path, "w", encoding=constants.CHARACTER_ENCODING) as f:
            f.write(request.files["text_file"].read().decode(constants.CHARACTER_ENCODING, "ignore").replace("\r\n", "\n"))
        request.files["audio_file"].save(audio_path)

        existing_output_path = os.path.join(output_folder, AUDIO_FOLDER)
        unlabelled_path = os.path.join(output_folder, UNLABELLED_FOLDER)
        existing_label_path = os.path.join(output_folder, METADATA_FILE)

        start_progress_thread(
            extend_existing_dataset,
            text_path=text_path,
            audio_path=audio_path,
            transcription_model=transcription_model,
            forced_alignment_path=forced_alignment_path,
            output_path=existing_output_path,
            unlabelled_path=unlabelled_path,
            label_path=existing_label_path,
            suffix=suffix,
            info_path=info_path,
            min_length=min_length,
            max_length=max_length,
            min_confidence=min_confidence,
            combine_clips=combine_clips,
        )

    return render_template("progress.html", next_url=get_next_url(constants.URLS, request.path))


@app.route("/dataset-duration", methods=["GET"])
def get_dataset_duration():
    dataset = request.values["dataset"]
    dataset_error = validate_dataset(
        os.path.join(paths["datasets"], dataset), metadata_file=METADATA_FILE, audio_folder=AUDIO_FOLDER
    )
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
    audio_folder = os.path.join(paths["datasets"], dataset_name, AUDIO_FOLDER)
    checkpoint_folder = os.path.join(paths["models"], dataset_name)

    if request.files.get("pretrained_model"):
        transfer_learning_path = os.path.join("data", "pretrained.pt")
        request.files["pretrained_model"].save(transfer_learning_path)
    else:
        transfer_learning_path = None

    start_progress_thread(
        train,
        metadata_path=metadata_path,
        dataset_directory=audio_folder,
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
        "progress.html", next_url=get_next_url(constants.URLS, request.path), voice=Path(checkpoint_folder).stem
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
        models=os.listdir(paths["models"]),
        checkpoints=get_checkpoints(),
        languages=get_languages(),
    )


@app.route("/synthesis-setup", methods=["POST"])
def synthesis_setup_post():
    global model, vocoder, symbols
    hifigan_folder = os.path.join(paths["hifigan"], request.form["vocoder"])
    model_path = os.path.join(hifigan_folder, "model.pt")
    model_config_path = os.path.join(hifigan_folder, "config.json")
    vocoder = Hifigan(model_path, model_config_path)
    dataset_name = request.form["model"]
    language = request.form["language"]
    symbols = get_symbols(language)
    checkpoint_folder = os.path.join(paths["models"], dataset_name)
    checkpoint = os.path.join(checkpoint_folder, request.form["checkpoint"])
    model = load_model(checkpoint)
    return redirect("/synthesis")


@app.route("/data/<path:path>")
def get_file(path):
    filename = path.split("/")[-1]
    mimetype = "image/png" if filename.endswith("png") else "audio/wav"
    return serve_file(os.path.join("data", path.replace("/", os.sep)), filename, mimetype)


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
        graph_path = os.path.join(results_folder, constants.GRAPH_FILE)
        audio_path = os.path.join(results_folder, constants.RESULTS_FILE)
        graph_web_path = graph_path.replace("\\", "/")
        audio_web_path = audio_path.replace("\\", "/")
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
    dataset.save(constants.TEMP_DATASET_UPLOAD)
    dataset_name = request.values["name"]
    dataset_directory = os.path.join(paths["datasets"], dataset_name)
    audio_folder = os.path.join(dataset_directory, AUDIO_FOLDER)
    assert not os.path.isdir(dataset_directory), "Output folder already exists"

    start_progress_thread(
        import_dataset, dataset=constants.TEMP_DATASET_UPLOAD, dataset_directory=dataset_directory, audio_folder=audio_folder
    )

    return render_template("progress.html", next_url="/import-export")


@app.route("/download-dataset", methods=["POST"])
def download_dataset():
    dataset_name = request.values["dataset"]
    dataset_directory = os.path.join(paths["datasets"], dataset_name)
    data = io.BytesIO()

    with zipfile.ZipFile(data, mode="w") as z:
        z.write(os.path.join(dataset_directory, METADATA_FILE), METADATA_FILE)
        if os.path.isfile(os.path.join(dataset_directory, INFO_FILE)):
            z.write(os.path.join(dataset_directory, INFO_FILE), INFO_FILE)

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
        request.files["model"].save(os.path.join(language_dir, constants.TRANSCRIPTION_MODEL))
    request.files["alphabet"].save(os.path.join(language_dir, constants.ALPHABET_FILE))
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
