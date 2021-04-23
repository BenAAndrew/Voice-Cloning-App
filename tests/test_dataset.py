import os
import shutil
import json
from pathlib import Path
from unittest import mock
import json

from dataset.create_dataset import create_dataset


expected_clips = {
    "0_2730.wav": "the examination and testimony of the experts",
    "2820_5100.wav": "enabled the commission to conclude",
    "5130_7560.wav": "that five shots may have been",
}
expected_duration = 7
expected_total_clips = 3


def fake_transcribe(path):
    filename = Path(path).name
    return expected_clips[filename]


def test_create_dataset():
    audio_path = os.path.join("tests", "files", "audio.wav")
    converted_audio_path = os.path.join("tests", "files", "audio-converted.wav")
    text_path = os.path.join("tests", "files", "text.txt")
    forced_alignment_path = "align.json"
    output_directory = "wavs"
    label_path = "metadata.csv"
    info_path = "info.json"
    min_confidence = 0.85

    with mock.patch("dataset.clip_generator.align.transcribe", wraps=fake_transcribe) as mock_transcribe:
        create_dataset(
            text_path=text_path,
            audio_path=audio_path,
            forced_alignment_path=forced_alignment_path,
            output_path=output_directory,
            label_path=label_path,
            info_path=info_path,
        )

    assert os.listdir(output_directory) == list(expected_clips.keys()), "Unexpected audio clips"

    with open(label_path) as f:
        lines = f.readlines()
        expected_text = [f"{name}|{text}\n" for name, text in expected_clips.items()]
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

    os.remove(forced_alignment_path)
    shutil.rmtree(output_directory)
    os.remove(label_path)
    os.remove(info_path)
    os.remove(converted_audio_path)
