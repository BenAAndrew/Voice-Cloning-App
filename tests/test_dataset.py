import os
import shutil
import json
from pathlib import Path
import json

from dataset.analysis import get_total_audio_duration, validate_dataset
from dataset.clip_generator import get_filename, extend_dataset
from dataset.create_dataset import create_dataset
from dataset.forced_alignment.search import similarity
from dataset.transcribe import create_transcription_model, TranscriptionModel, DeepSpeech, Silero


TEXT = "the examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"
EXPECTED_CLIPS = {
    "0_2730.wav": "the examination and testimony of the experts",
    "2820_5100.wav": "enabled the commission to conclude",
    "5130_7560.wav": "that five shots may have been",
}


class FakeTranscriptionModel(TranscriptionModel):
    def load_audio(self, path):
        pass

    def transcribe(self, path):
        filename = Path(path).name
        return EXPECTED_CLIPS[filename]


# Dataset creation
def test_create_dataset():
    audio_path = os.path.join("test_samples", "audio.wav")
    converted_audio_path = os.path.join("test_samples", "audio-converted.wav")
    text_path = os.path.join("test_samples", "text.txt")
    dataset_directory = "test-dataset"
    forced_alignment_path = os.path.join(dataset_directory, "align.json")
    output_directory = os.path.join(dataset_directory, "wavs")
    label_path = os.path.join(dataset_directory, "metadata.csv")
    info_path = os.path.join(dataset_directory, "info.json")
    min_confidence = 0.85

    create_dataset(
        text_path=text_path,
        audio_path=audio_path,
        transcription_model=FakeTranscriptionModel(),
        forced_alignment_path=forced_alignment_path,
        output_path=output_directory,
        label_path=label_path,
        info_path=info_path,
    )

    assert os.listdir(output_directory) == list(EXPECTED_CLIPS.keys()), "Unexpected audio clips"

    with open(label_path) as f:
        lines = f.readlines()
        expected_text = [f"{name}|{text}\n" for name, text in EXPECTED_CLIPS.items()]
        assert lines == expected_text, "Unexpected metadata contents"

    with open(forced_alignment_path, "r") as forced_alignment_file:
        data = json.load(forced_alignment_file)
        for segment in data:
            assert {"start", "end", "name", "score", "aligned"}.issubset(
                segment.keys()
            ), "Alignment JSON missing required keys"
            assert segment["score"] >= min_confidence, "SWS score less than min confidence"

    with open(info_path) as f:
        data = json.load(f)
        assert int(data["total_duration"]) == 7
        assert data["total_clips"] == 3

    os.remove(converted_audio_path)
    shutil.rmtree(dataset_directory)


# Extend dataset
def test_extend_dataset():
    dataset_directory = "test-dataset"
    audio_folder = os.path.join(dataset_directory, "wavs")
    metadata_file = os.path.join(dataset_directory, "metadata.csv")
    os.makedirs(dataset_directory)
    os.makedirs(audio_folder)
    with open(metadata_file, "w") as f:
        pass

    audio_path = os.path.join("test_samples", "audio.wav")
    text_path = os.path.join("test_samples", "text.txt")
    forced_alignment_path = os.path.join(dataset_directory, "align.json")
    label_path = os.path.join(dataset_directory, "metadata.csv")
    suffix = "extend"
    extend_dataset(
        audio_path=audio_path,
        script_path=text_path,
        transcription_model=FakeTranscriptionModel(),
        forced_alignment_path=forced_alignment_path,
        output_path=audio_folder,
        label_path=label_path,
        suffix=suffix,
    )

    assert os.listdir(audio_folder) == [name.split('.')[0]+"-"+suffix+".wav" for name in EXPECTED_CLIPS.keys()], "Unexpected audio clips"

    with open(label_path) as f:
        lines = f.readlines()
        expected_text = [f"{name.split('.')[0]}-{suffix}.wav|{text}\n" for name, text in EXPECTED_CLIPS.items()]
        assert lines == expected_text, "Unexpected metadata contents"

    shutil.rmtree(dataset_directory)


def test_get_filename():
    new_filename = get_filename("audio.wav", "converted")
    assert new_filename == "audio-converted.wav"


# Analysis
def test_get_total_audio_duration():
    info_path = os.path.join("test_samples", "info.json")
    duration, total_clips = get_total_audio_duration(info_path)
    assert duration == 10000
    assert total_clips == 100


def test_validate_dataset():
    output_directory = "test-dataset"
    os.makedirs(output_directory)

    # No files
    message = validate_dataset(output_directory)
    assert message == "Missing metadata.csv file"

    # No info or wavs
    with open(os.path.join(output_directory, "metadata.csv"), "w") as f:
        pass
    message = validate_dataset(output_directory)
    assert message == "Missing info.json file"

    # No wavs
    with open(os.path.join(output_directory, "info.json"), "w") as f:
        pass
    message = validate_dataset(output_directory)
    assert message == "Missing wavs folder"

    # All required files
    os.makedirs(os.path.join(output_directory, "wavs"))
    message = validate_dataset(output_directory)
    assert not message

    shutil.rmtree(output_directory)


# Transcription
def test_deepspeech():
    model_path = os.path.join("test_samples", "english.pbmm")
    deepspeech = create_transcription_model(model_path)
    assert isinstance(deepspeech, DeepSpeech)

    audio_path = os.path.join("test_samples", "audio.wav")
    transcription = deepspeech.transcribe(audio_path)
    assert similarity(TEXT, transcription) > 0.5


def test_silero():
    silero = create_transcription_model()
    assert isinstance(silero, Silero)

    audio_path = os.path.join("test_samples", "audio.wav")
    transcription = silero.transcribe(audio_path)
    assert similarity(TEXT, transcription) > 0.5
