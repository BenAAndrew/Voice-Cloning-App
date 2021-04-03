import os
import shutil
import json
from pathlib import Path
from unittest import mock

from dataset.clip_generator import clip_generator


expected_clips = {
    "0_2730.wav": "the examination and testimony of the experts",
    "2820_5100.wav": "enabled the commission to conclude",
    "5130_7560.wav": "that five shots may have been fired"
}


def fake_transcribe(path):
    filename = Path(path).name
    return expected_clips[filename]


def test_clip_generator():
    audio_path = os.path.join("tests", "files", "audio.wav")
    script_path = os.path.join("tests", "files", "text.txt")
    forced_alignment_path = "align.json"
    output_directory = "wavs"
    label_path = "metadata.csv"
    min_confidence = 0.85

    with mock.patch("dataset.clip_generator.align.transcribe", wraps=fake_transcribe) as mock_transcribe:
        clip_generator(
            audio_path=audio_path,
            script_path=script_path,
            forced_alignment_path=forced_alignment_path,
            output_path=output_directory,
            label_path=label_path,
            min_confidence=min_confidence,
        )

    assert os.listdir(output_directory) == list(expected_clips.keys()), "Unexpected audio clips"

    with open(label_path) as f:
        lines = f.readlines()
        expected_text = [f"{name}|{text}\n" for name, text in expected_clips.items()]
        assert lines == expected_text, "Unexpected metadata contents"

    with open(forced_alignment_path, "r") as forced_alignment_file:
        data = json.load(forced_alignment_file)
        for segment in data:
            assert {"start", "end", "name", "sws", "aligned"}.issubset(
                segment.keys()
            ), "Alignment JSON missing required keys"
            assert segment["sws"] >= min_confidence, "SWS score less than min confidence"

    os.remove(forced_alignment_path)
    shutil.rmtree(output_directory)
    os.remove(label_path)
