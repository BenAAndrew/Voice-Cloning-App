import os
import inflect
import random
from string import ascii_lowercase
from unittest import mock
import torch
import shutil

from dataset.clip_generator import CHARACTER_ENCODING
from training.clean_text import clean_text
from training.checkpoint import load_checkpoint, save_checkpoint, warm_start_model
from training.dataset import VoiceDataset
from training.tacotron2_model import Tacotron2
from training.train import DEFAULT_ALPHABET, WEIGHT_DECAY
from training.utils import (
    check_space,
    load_metadata,
    load_symbols,
    get_learning_rate,
    get_batch_size,
    LEARNING_RATE_PER_BATCH,
    BATCH_SIZE_PER_GB,
    CHECKPOINT_SIZE_MB,
    PUNCTUATION,
)


# Clean text
def test_clean_text():
    text = clean_text("1st $500 Mr. 10.5 2,000 30 a\tb ~", inflect.engine())
    assert text == "first five hundred dollars mister ten point five two thousand thirty a b "


# Dataset
def test_voice_dataset():
    random.seed(1234)

    metadata_path = os.path.join("test_samples", "dataset", "metadata.csv")
    audio_directory = os.path.join("test_samples", "dataset", "wavs")
    with open(metadata_path, encoding=CHARACTER_ENCODING) as f:
        filepaths_and_text = [line.strip().split("|") for line in f]

    dataset = VoiceDataset(filepaths_and_text, audio_directory, DEFAULT_ALPHABET)
    assert len(dataset) == 3
    text, mel = dataset[0]
    assert len(text) == len("that five shots may have been")
    assert mel.shape[0] == 80


# Checkpoints
def test_load_and_save_checkpoint():
    model_path = os.path.join("test_samples", "model.pt")
    model = Tacotron2()
    lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model, optimizer, iteration, epoch = load_checkpoint(model_path, model, optimizer, [None] * 10000)

    assert iteration == 510000
    assert epoch == iteration // 10000

    checkpoint_folder = "test-checkpoints"
    os.makedirs(checkpoint_folder)
    save_checkpoint(model, optimizer, lr, iteration, epoch, checkpoint_folder)
    assert "checkpoint_510000" in os.listdir(checkpoint_folder)

    # Overwrite existing
    save_checkpoint(model, optimizer, lr, 520000, epoch, checkpoint_folder, overwrite_checkpoints=True)
    assert "checkpoint_520000" in os.listdir(checkpoint_folder)
    assert "checkpoint_510000" not in os.listdir(checkpoint_folder)

    # Do not overwrite existing
    save_checkpoint(model, optimizer, lr, 530000, epoch, checkpoint_folder, overwrite_checkpoints=False)
    assert "checkpoint_520000" in os.listdir(checkpoint_folder)
    assert "checkpoint_530000" in os.listdir(checkpoint_folder)

    shutil.rmtree(checkpoint_folder)


def test_warm_start_model():
    model_path = os.path.join("test_samples", "model.pt")
    model = Tacotron2()
    ignore_layers = ["embedding.weight"]
    model = warm_start_model(model_path, model, ignore_layers)
    model_dict = model.state_dict()

    checkpoint_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    for k in checkpoint_dict.keys():
        if k not in ignore_layers:
            assert torch.equal(model_dict[k], checkpoint_dict[k])


# Metadata
def test_load_metadata():
    metadata_path = os.path.join("test_samples", "dataset", "metadata.csv")
    data = {
        "0_2730.wav": "the examination and testimony of the experts",
        "2820_5100.wav": "enabled the commission to conclude",
        "5130_7560.wav": "that five shots may have been",
    }
    train_size = 0.67
    train_files, test_files = load_metadata(metadata_path, train_size)
    assert len(train_files) == 2
    for name, text in train_files:
        assert data[name] == text

    assert len(test_files) == 1
    name, text = test_files[0]
    assert data[name] == text
    assert name not in [i[0] for i in train_files]


# Symbols
def test_load_symbols():
    alphabet_path = os.path.join("test_samples", "english.txt")
    symbols = set(load_symbols(alphabet_path))
    assert set(ascii_lowercase).issubset(symbols)
    assert set(PUNCTUATION).issubset(symbols)


# Disk usage
@mock.patch("shutil.disk_usage")
def test_check_space_failure(disk_usage):
    disk_usage.return_value = None, None, (CHECKPOINT_SIZE_MB) * (2 ** 20)
    exception = False
    try:
        check_space(2)
    except Exception as e:
        exception = True
        assert type(e) == AssertionError
    assert exception, "Insufficent space should throw an exception"


@mock.patch("shutil.disk_usage")
def test_check_space_success(disk_usage):
    disk_usage.return_value = None, None, (CHECKPOINT_SIZE_MB + 1) * (2 ** 20)
    assert check_space(1) is None, "Sufficent space should not throw an exception"


# Test parameters
def test_get_learning_rate():
    batch_size = 40
    lr = get_learning_rate(batch_size)
    assert lr == batch_size * LEARNING_RATE_PER_BATCH


def test_get_batch_size():
    memory = 8
    batch_size = get_batch_size(memory)
    assert batch_size == int(memory * BATCH_SIZE_PER_GB)
