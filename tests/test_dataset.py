import os
import shutil
import json
from pathlib import Path
import json
import pysrt

from tests.test_synthesis import MIN_SYNTHESIS_SCORE
from dataset.analysis import get_total_audio_duration, get_clip_lengths, validate_dataset, get_text, update_dataset_info
from dataset.clip_generator import generate_clips_from_subtitles, clip_combiner
from dataset.create_dataset import create_dataset
from dataset.extend_existing_dataset import extend_existing_dataset
from dataset.utils import similarity, add_suffix, get_invalid_characters
from dataset.transcribe import TranscriptionModel, DeepSpeech, Silero


TEXT = "the examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"
TRANSCRIPTION = {
    "000000000_000002730.wav": "the examination and testimony of the experts",
    "000002820_000005100.wav": "enabled the commission to conclude",
    "000005130_000007560.wav": "that five shots may have been fired",
}
EXPECTED_CLIPS = ["000000000_000002730.wav", "000002820_000005100.wav"]
UNMATCHED_CLIPS = ["000005130_000007560.wav"]
EXPECTED_SUBTITLE_CLIPS = {
    "000000000000_000002600000.wav": "The examination and testimony of the experts",
    "000002900000_000007400000.wav": "enabled the Commission to conclude that five shots may have been fired,",
}


class FakeTranscriptionModel(TranscriptionModel):
    def load_audio(self, path):
        pass

    def transcribe(self, path):
        filename = Path(path).name
        return TRANSCRIPTION[filename]


# Invalid characters
def test_get_invalid_characters():
    invalid_chars = get_invalid_characters("aà1!", ["a"])
    assert invalid_chars == set("à")


# Dataset creation
def test_create_dataset():
    audio_path = os.path.join("test_samples", "audio.wav")
    converted_audio_path = os.path.join("test_samples", "audio-converted.wav")
    text_path = os.path.join("test_samples", "text.txt")
    dataset_directory = "test-create-dataset"
    forced_alignment_path = os.path.join(dataset_directory, "align.json")
    output_directory = os.path.join(dataset_directory, "wavs")
    unlabelled_path = os.path.join(dataset_directory, "unlabelled")
    label_path = os.path.join(dataset_directory, "metadata.csv")
    info_path = os.path.join(dataset_directory, "info.json")
    min_confidence = 1.0

    create_dataset(
        text_path=text_path,
        audio_path=audio_path,
        transcription_model=FakeTranscriptionModel(),
        output_folder=dataset_directory,
        min_confidence=min_confidence,
        combine_clips=False,
    )

    assert os.listdir(output_directory) == EXPECTED_CLIPS, "Unexpected audio clips"
    assert os.listdir(unlabelled_path) == UNMATCHED_CLIPS, "Unexpected unmatched audio clips"

    with open(label_path) as f:
        lines = f.readlines()
        expected_text = [f"{name}|{text}\n" for name, text in TRANSCRIPTION.items() if name in EXPECTED_CLIPS]
        assert lines == expected_text, "Unexpected metadata contents"

    with open(forced_alignment_path, "r") as forced_alignment_file:
        data = json.load(forced_alignment_file)
        for segment in data:
            assert {"name", "start", "end", "duration", "score", "transcript", "text"}.issubset(
                segment.keys()
            ), "Alignment JSON missing required keys"
            assert segment["score"] >= min_confidence, "SWS score less than min confidence"

    with open(info_path) as f:
        data = json.load(f)
        assert int(data["total_duration"]) == 5
        assert data["total_clips"] == 2

    os.remove(converted_audio_path)
    shutil.rmtree(dataset_directory)


class FakeSubtitleTranscriptionModel(TranscriptionModel):
    def load_audio(self, path):
        pass

    def transcribe(self, path):
        filename = Path(path).name
        return EXPECTED_SUBTITLE_CLIPS[filename]


def test_generate_clips_from_subtitles():
    dataset_directory = "test-subtitles"
    os.makedirs(dataset_directory)
    audio_path = os.path.join("test_samples", "audio.wav")
    subtitle_path = os.path.join("test_samples", "sub.srt")
    subs = pysrt.open(subtitle_path)

    clips, unlabelled_clips, clip_lengths = generate_clips_from_subtitles(
        audio_path=audio_path,
        subs=subs,
        transcription_model=FakeSubtitleTranscriptionModel(),
        output_path=dataset_directory,
    )

    assert clips == [
        {
            "name": "000000000000_000002600000.wav",
            "start": "00:00:00.000000",
            "end": "00:00:02.600000",
            "transcript": "The examination and testimony of the experts",
            "text": "The examination and testimony of the experts",
            "score": 1,
            "duration": 2.6,
        },
        {
            "name": "000002900000_000007400000.wav",
            "start": "00:00:02.900000",
            "end": "00:00:07.400000",
            "transcript": "enabled the Commission to conclude that five shots may have been fired,",
            "text": "enabled the Commission to conclude that five shots may have been fired,",
            "score": 1,
            "duration": 4.5,
        },
    ]
    assert unlabelled_clips == []
    assert clip_lengths == [2.6, 4.5]

    shutil.rmtree(dataset_directory)


# Clip combiner
def test_clip_combiner():
    output_directory = "test-clip-combiner"
    audio_path = os.path.join("test_samples", "audio.wav")
    os.makedirs(output_directory)

    clips = [
        # Should combine first 2 (duration would be 8 seconds)
        {
            "name": "000000000_000002000.wav",
            "start": "00:00:00.000",
            "end": "00:00:02.000",
            "duration": 2,
            "transcript": "abcd",
            "text": "abc",
            "score": 0.8,
        },
        {
            "name": "000004000_000008000.wav",
            "start": "00:00:04.000",
            "end": "00:00:08.000",
            "duration": 4,
            "transcript": "xyz",
            "text": "xyz",
            "score": 1.0,
        },
        # Shouldn't combine last 2 (duration would be 12 seconds)
        {
            "name": "000010000_000015000.wav",
            "start": "00:00:10.000",
            "end": "00:00:15.000",
            "duration": 5,
            "transcript": "apple",
            "text": "apple",
            "score": 1.0,
        },
        {
            "name": "000017000_000022000.wav",
            "start": "00:00:17.000",
            "end": "00:00:22.000",
            "duration": 5,
            "transcript": "banana",
            "text": "banana",
            "score": 1.0,
        },
    ]

    clips, clip_lengths = clip_combiner(audio_path, output_directory, clips, max_length=10)

    assert clips == [
        {
            "name": "000000000_000008000.wav",
            "start": "00:00:00.000",
            "end": "00:00:08.000",
            "duration": 8,
            "transcript": "abcd, xyz",
            "text": "abc, xyz",
            "score": 0.9,
        },
        {
            "name": "000010000_000015000.wav",
            "start": "00:00:10.000",
            "end": "00:00:15.000",
            "duration": 5,
            "transcript": "apple",
            "text": "apple",
            "score": 1.0,
        },
        {
            "name": "000017000_000022000.wav",
            "start": "00:00:17.000",
            "end": "00:00:22.000",
            "duration": 5,
            "transcript": "banana",
            "text": "banana",
            "score": 1.0,
        },
    ]

    shutil.rmtree(output_directory)


# Extend dataset
def test_extend_existing_dataset():
    dataset_directory = "test-extend-dataset"
    audio_folder = os.path.join(dataset_directory, "wavs")
    unlabelled_folder = os.path.join(dataset_directory, "unlabelled")
    metadata_file = os.path.join(dataset_directory, "metadata.csv")
    os.makedirs(dataset_directory)
    os.makedirs(audio_folder)
    os.makedirs(unlabelled_folder)
    with open(metadata_file, "w") as f:
        pass

    audio_path = os.path.join("test_samples", "audio.wav")
    converted_audio_path = os.path.join("test_samples", "audio-converted.wav")
    text_path = os.path.join("test_samples", "text.txt")
    label_path = os.path.join(dataset_directory, "metadata.csv")
    suffix = "extend"
    min_confidence = 1.0
    extend_existing_dataset(
        text_path=text_path,
        audio_path=audio_path,
        transcription_model=FakeTranscriptionModel(),
        output_folder=dataset_directory,
        suffix=suffix,
        min_confidence=min_confidence,
        combine_clips=False,
    )

    assert os.listdir(audio_folder) == [
        name.split(".")[0] + "-" + suffix + ".wav" for name in EXPECTED_CLIPS
    ], "Unexpected audio clips"

    assert os.listdir(unlabelled_folder) == [
        name.split(".")[0] + "-" + suffix + ".wav" for name in UNMATCHED_CLIPS
    ], "Unexpected unlabelled audio clips"

    with open(label_path) as f:
        lines = f.readlines()
        expected_text = [
            f"{name.split('.')[0]}-{suffix}.wav|{text}\n"
            for name, text in TRANSCRIPTION.items()
            if name in EXPECTED_CLIPS
        ]
        assert lines == expected_text, "Unexpected metadata contents"

    os.remove(converted_audio_path)
    shutil.rmtree(dataset_directory)


# Utils
def test_add_suffix():
    new_filename = add_suffix("audio.wav", "converted")
    assert new_filename == "audio-converted.wav"


def test_similarity():
    assert similarity("abc", "def") == 0
    assert similarity("abc", "abc") == 1


# Analysis
def test_get_total_audio_duration():
    info_path = os.path.join("test_samples", "info.json")
    data = get_total_audio_duration(info_path)
    assert data["total_duration"] == 10000
    assert data["total_clips"] == 100


def test_get_clip_lengths():
    folder = os.path.join("test_samples", "dataset", "wavs")
    clips_lengths = get_clip_lengths(folder)
    assert clips_lengths == [2.8299319727891157, 2.379909297052154, 2.529931972789116]


def test_validate_dataset():
    output_directory = "test-validate"
    os.makedirs(output_directory)

    # No files
    message = validate_dataset(output_directory)
    assert message == "Missing metadata.csv or trainlist.txt/vallist.txt file"

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


def test_update_dataset_info():
    info_path = os.path.join("test_samples", "info.json")
    shutil.copy(info_path, os.path.join("test_samples", "info-backup.json"))
    metadata_path = os.path.join("test_samples", "dataset", "metadata.csv")
    clip_path = os.path.join("test_samples", "dataset", "wavs", "0_2730.wav")
    text = "some text"

    with open(info_path) as f:
        info_before = json.load(f)

    update_dataset_info(metadata_path, info_path, clip_path, text)

    with open(info_path) as f:
        data = json.load(f)
        assert data["total_duration"] == info_before["total_duration"] + 2.829931972788
        assert data["total_clips"] == info_before["total_clips"] + 1

    os.remove(info_path)
    shutil.copy(os.path.join("test_samples", "info-backup.json"), info_path)


# Transcription
def test_deepspeech():
    model_path = os.path.join("test_samples", "english.pbmm")
    transcription_model = DeepSpeech(model_path)

    audio_path = os.path.join("test_samples", "audio.wav")
    transcription = transcription_model.transcribe(audio_path)
    assert similarity(TEXT, transcription) > MIN_SYNTHESIS_SCORE


def test_silero():
    transcription_model = Silero()

    audio_path = os.path.join("test_samples", "audio.wav")
    transcription = transcription_model.transcribe(audio_path)
    assert similarity(TEXT, transcription) > MIN_SYNTHESIS_SCORE
