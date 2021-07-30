import os
import inflect
import random
from string import ascii_lowercase
from unittest import mock
import torch
from torch.utils.data import DataLoader
import shutil

from dataset.clip_generator import CHARACTER_ENCODING
from training.clean_text import clean_text
from training.checkpoint import load_checkpoint, save_checkpoint, warm_start_model
from training.dataset import VoiceDataset
from training.tacotron2_model import Tacotron2, TextMelCollate
from training.train import train, MINIMUM_MEMORY_GB, DEFAULT_ALPHABET, WEIGHT_DECAY
from training.utils import (
    check_space,
    load_metadata,
    load_symbols,
    get_learning_rate,
    get_batch_size,
    check_early_stopping,
    LEARNING_RATE_PER_BATCH,
    BATCH_SIZE_PER_GB,
    CHECKPOINT_SIZE_MB,
    PUNCTUATION,
)


# Training
class MockedTacotron2:
    _state_dict = {"param": None}

    def cuda():
        return MockedTacotron2()

    def parameters(self):
        return {}

    def train(self):
        pass

    def zero_grad(self):
        pass

    def state_dict(self):
        return self._state_dict


class MockedTacotron2Loss:
    def __init__(self, y_pred, y):
        pass

    def item(self):
        return 0.5

    def backward(self):
        pass


class MockedOptimizer:
    param_groups = [{"lr": 0.1}]
    _state_dict =  {"lr": 0.1}

    def __init__(self, parameters, lr, weight_decay):
        pass

    def step():
        pass

    def state_dict():
        return MockedOptimizer._state_dict


@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("training.train.get_available_memory", return_value=MINIMUM_MEMORY_GB)
@mock.patch("training.train.Tacotron2", return_value=MockedTacotron2)
@mock.patch("training.train.Tacotron2Loss", return_value=MockedTacotron2Loss)
@mock.patch("torch.optim.Adam", return_value=MockedOptimizer)
@mock.patch("training.train.process_batch", return_value=(None, None))
@mock.patch("training.train.validate", return_value=0.6)
def test_training_a(validate, process_batch, Adam, Tacotron2Loss, Tacotron2, get_available_memory, is_available):    
    metadata_path = os.path.join("test_samples", "dataset", "metadata.csv")
    dataset_directory = os.path.join("test_samples", "dataset", "wavs")
    output_directory = "checkpoint"
    train_size=0.67

    train(
        metadata_path,
        dataset_directory,
        output_directory,
        epochs=1,
        batch_size=1,
        early_stopping=False,
        multi_gpu=False,
        train_size=train_size
    )

    assert is_available.called
    assert get_available_memory.called
    assert Tacotron2.called
    assert Tacotron2Loss.called
    assert Adam.called

    with open(metadata_path, encoding=CHARACTER_ENCODING) as f:
        data = [line.strip().split("|") for line in f]
    dataset = VoiceDataset(data, dataset_directory, DEFAULT_ALPHABET)
    collate_fn = TextMelCollate()
    data_loader = DataLoader(
        dataset, num_workers=0, sampler=None, batch_size=1, pin_memory=False, collate_fn=collate_fn
    )

    # Check batches are equal
    assert len(process_batch.mock_calls) == 2
    called_batches = [call[1][0] for call in process_batch.mock_calls]
    batches = [b for b in data_loader]
    batch_sizes = [b[0].size() for b in batches]

    for called_batch in called_batches:
        index = batch_sizes.index(called_batch[0].size())
        for i in range(len(called_batch)):
            assert torch.equal(batches[index][i], called_batch[i])

    # Check validate iterations called
    assert len(validate.mock_calls) == 2
    iterations_called = [call[1][3] for call in validate.mock_calls]
    assert iterations_called[0] == 0
    assert iterations_called[1] == 2

    # Check checkpoint
    checkpoint_path = os.path.join(output_directory, "checkpoint_2")
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint_dict["state_dict"] == MockedTacotron2._state_dict
    assert checkpoint_dict["optimizer"] == MockedOptimizer._state_dict
    assert checkpoint_dict["iteration"] == 2
    assert checkpoint_dict["epoch"] == 1

    shutil.rmtree(output_directory)


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


# Early stopping
def test_early_stopping():
    # Too few values
    assert check_early_stopping([10,10,10,10]) is False

    # Loss still improving
    assert check_early_stopping([1.1,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]) is False

    # Loss not improving
    assert check_early_stopping([0.5,0.4999,0.5,0.4999,0.5,0.4998,0.4999,0.4996,0.4997,0.5]) is True


# Disk usage
@mock.patch("shutil.disk_usage", return_value=(None, None, (CHECKPOINT_SIZE_MB) * (2 ** 20)))
def test_check_space_failure(disk_usage):
    exception = False
    try:
        check_space(2)
    except Exception as e:
        exception = True
        assert type(e) == AssertionError
    assert exception, "Insufficent space should throw an exception"


@mock.patch("shutil.disk_usage", return_value=(None, None, (CHECKPOINT_SIZE_MB + 1) * (2 ** 20)))
def test_check_space_success(disk_usage):
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
