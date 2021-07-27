import os
import sys
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
    get_suffix,
    delete_folder,
    import_dataset,
)
from dataset.create_dataset import create_dataset
from dataset.clip_generator import CHARACTER_ENCODING
from dataset.extend_existing_dataset import extend_existing_dataset
from dataset.analysis import get_total_audio_duration, validate_dataset
from dataset.transcribe import create_transcription_model
from training.train import train, DEFAULT_ALPHABET
from training.utils import get_available_memory, get_batch_size, load_symbols
from synthesis.synthesize import load_model, synthesize
from synthesis.waveglow import Waveglow
from synthesis.hifigan import Hifigan

from flask import redirect, render_template, request, send_file


URLS = {"/": "Build dataset", "/train": "Train", "/synthesis-setup": "Synthesis"}
TEXT_FILE = "text.txt"
ALIGNMENT_FILE = "align.json"
AUDIO_FOLDER = "wavs"
METADATA_FILE = "metadata.csv"
INFO_FILE = "info.json"
CHECKPOINT_FOLDER = "checkpoints"
GRAPH_FILE = "graph.png"
RESULTS_FILE = "out.wav"
TEMP_DATASET_UPLOAD = "temp.zip"
TRANSCRIPTION_MODEL = "model.pbmm"
ALPHABET_FILE = "alphabet.txt"
ENGLISH_LANGUAGE = "English"

model = None
vocoder = None
inflect_engine = inflect.engine()
symbols = None


def get_languages():
    return [ENGLISH_LANGUAGE] + os.listdir(paths["languages"])


def get_checkpoints():
    # Checkpoints ordered by name (i.e. checkpoint_0, checkpoint_1000 etc.)
    return {
        model: sorted(
            os.listdir(os.path.join(paths["models"], model)),
            key=lambda name: int(name.split("_")[1]) if "_" in name and name.split("_")[1].isdigit() else 0,
            reverse=True,
        )
        for model in os.listdir(paths["models"])
    }


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


@app.route("/datasource", methods=["GET"])
def get_datasource():
    return render_template("datasource.html")


@app.route("/", methods=["POST"])
def create_dataset_post():
    min_confidence = float(request.form["confidence"])
    language = request.form["language"]
    transcription_model_path = (
        os.path.join(paths["languages"], language, TRANSCRIPTION_MODEL) if language != ENGLISH_LANGUAGE else None
    )
    transcription_model = create_transcription_model(transcription_model_path)

    if request.form["name"]:
        output_folder = os.path.join(paths["datasets"], request.form["name"])
        if os.path.exists(output_folder):
            request.files = None
            raise Exception("Dataset name taken")

        os.makedirs(output_folder, exist_ok=True)
        text_path = os.path.join(output_folder, TEXT_FILE)
        audio_path = os.path.join(output_folder, request.files["audio_file"].filename)
        forced_alignment_path = os.path.join(output_folder, ALIGNMENT_FILE)
        output_path = os.path.join(output_folder, AUDIO_FOLDER)
        label_path = os.path.join(output_folder, METADATA_FILE)
        info_path = os.path.join(output_folder, INFO_FILE)

        with open(text_path, "w", encoding=CHARACTER_ENCODING) as f:
            f.write(request.files["text_file"].read().decode(CHARACTER_ENCODING, "ignore").replace("\r\n", "\n"))
        request.files["audio_file"].save(audio_path)

        start_progress_thread(
            create_dataset,
            text_path=text_path,
            audio_path=audio_path,
            transcription_model=transcription_model,
            forced_alignment_path=forced_alignment_path,
            output_path=output_path,
            label_path=label_path,
            info_path=info_path,
            min_confidence=min_confidence,
        )
    else:
        output_folder = os.path.join(paths["datasets"], request.form["dataset"])
        suffix = get_suffix()
        text_path = os.path.join(output_folder, f"text-{suffix}.txt")
        audio_path = os.path.join(output_folder, f"audio-{suffix}.mp3")
        forced_alignment_path = os.path.join(output_folder, f"align-{suffix}.json")
        info_path = os.path.join(output_folder, INFO_FILE)

        with open(text_path, "w", encoding=CHARACTER_ENCODING) as f:
            f.write(request.files["text_file"].read().decode(CHARACTER_ENCODING, "ignore").replace("\r\n", "\n"))
        request.files["audio_file"].save(audio_path)

        existing_output_path = os.path.join(output_folder, AUDIO_FOLDER)
        existing_label_path = os.path.join(output_folder, METADATA_FILE)

        start_progress_thread(
            extend_existing_dataset,
            text_path=text_path,
            audio_path=audio_path,
            transcription_model=transcription_model,
            forced_alignment_path=forced_alignment_path,
            output_path=existing_output_path,
            label_path=existing_label_path,
            suffix=suffix,
            info_path=info_path,
            min_confidence=min_confidence,
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
    alphabet_path = os.path.join(paths["languages"], language, ALPHABET_FILE) if language != ENGLISH_LANGUAGE else None
    dataset_name = request.form["dataset"]
    epochs = request.form["epochs"]
    batch_size = request.form["batch_size"]
    early_stopping = request.form.get("early_stopping") is not None
    iters_per_checkpoint = request.form["checkpoint_frequency"]
    overwrite_checkpoints = request.form.get("overwrite_checkpoints") is not None
    multi_gpu = request.form.get("multi_gpu") is not None
    checkpoint_path = (
        os.path.join(paths["models"], dataset_name, request.form["checkpoint"]) if request.form.get("checkpoint") else None
    )

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

    start_progress_thread(
        train,
        metadata_path=metadata_path,
        dataset_directory=audio_folder,
        output_directory=checkpoint_folder,
        alphabet_path=alphabet_path,
        checkpoint_path=checkpoint_path,
        transfer_learning_path=transfer_learning_path,
        epochs=int(epochs),
        batch_size=int(batch_size),
        early_stopping=early_stopping,
        multi_gpu=multi_gpu,
        overwrite_checkpoints=overwrite_checkpoints,
        iters_per_checkpoint=int(iters_per_checkpoint),
    )

    return render_template("progress.html", next_url=get_next_url(URLS, request.path))


# Synthesis
@app.route("/synthesis-setup", methods=["GET"])
def get_synthesis_setup():
    return render_template(
        "synthesis-setup.html",
        waveglow_models=os.listdir(paths["waveglow"]),
        hifigan_models=os.listdir(paths["hifigan"]),
        models=os.listdir(paths["models"]),
        checkpoints=get_checkpoints(),
        languages=get_languages(),
    )


@app.route("/synthesis-setup", methods=["POST"])
def synthesis_setup_post():
    global model, vocoder, symbols
    vocoder_type = request.form["vocoder_type"]
    if vocoder_type == "hifigan":
        hifigan_folder = os.path.join(paths["hifigan"], request.form["vocoder"])
        model_path = os.path.join(hifigan_folder, "model.pt")
        model_config_path = os.path.join(hifigan_folder, "config.json")
        vocoder = Hifigan(model_path, model_config_path)
    elif vocoder_type == "waveglow":
        model_path = os.path.join(paths["waveglow"], request.form["vocoder"])
        vocoder = Waveglow(model_path)
    else:
        return render_template("synthesis-setup.html", error="Invalid vocoder selected")

    dataset_name = request.form["model"]
    language = request.form["language"]
    alphabet_path = os.path.join(paths["languages"], language, ALPHABET_FILE)
    symbols = load_symbols(alphabet_path) if language != ENGLISH_LANGUAGE else DEFAULT_ALPHABET
    checkpoint_folder = os.path.join(paths["models"], dataset_name)
    checkpoint = os.path.join(checkpoint_folder, request.form["checkpoint"])
    model = load_model(checkpoint)
    return redirect("/synthesis")


@app.route("/data/results/<path:path>")
def get_result_file(path):
    filename = path.split("/")[-1]
    mimetype = "image/png" if filename.endswith("png") else "audio/wav"

    with open(os.path.join(paths["results"], path), "rb") as f:
        return send_file(io.BytesIO(f.read()), attachment_filename=filename, mimetype=mimetype, as_attachment=True)


@app.route("/synthesis", methods=["GET", "POST"])
def synthesis_post():
    global model, vocoder, symbols
    if not model or not vocoder or not symbols:
        return redirect("/synthesis-setup")

    if request.method == "GET":
        return render_template("synthesis.html")
    else:
        text = request.form["text"]
        folder_name = get_suffix()
        results_folder = os.path.join(paths["results"], folder_name)
        os.makedirs(results_folder)
        graph_path = os.path.join(results_folder, GRAPH_FILE)
        audio_path = os.path.join(results_folder, RESULTS_FILE)
        graph_web_path = graph_path.replace("\\", "/")
        audio_web_path = audio_path.replace("\\", "/")

        synthesize(model, text, inflect_engine, symbols, graph_path, audio_path, vocoder)
        return render_template(
            "synthesis.html",
            text=text.strip(),
            graph=graph_web_path,
            audio=audio_web_path,
        )


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
    request.files["model"].save(os.path.join(language_dir, TRANSCRIPTION_MODEL))
    request.files["alphabet"].save(os.path.join(language_dir, ALPHABET_FILE))
    return redirect("/settings")


@app.route("/add-vocoder", methods=["POST"])
def add_vocoder():
    vocoder_type = request.values["vocoder"]
    name = request.values["name"]

    if vocoder_type == "hifigan":
        hifigan_folder = os.path.join(paths["hifigan"], name)
        os.makedirs(hifigan_folder)
        model_path = os.path.join(hifigan_folder, "model.pt")
        model_config_path = os.path.join(hifigan_folder, "config.json")
        request.files["hifigan-model"].save(model_path)
        request.files["hifigan-config"].save(model_config_path)
    else:
        model_path = os.path.join(paths["waveglow"], name+".pt")
        request.files["waveglow"].save(model_path)

    return redirect("/settings")
