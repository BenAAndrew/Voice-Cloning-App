import os
import inflect
import random
from string import ascii_lowercase
from unittest import mock
import torch
import shutil

from dataset.clip_generator import CHARACTER_ENCODING
from training.clean_text import clean_text
from training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    checkpoint_cleanup,
    warm_start_model,
    transfer_symbols_embedding,
)
from training.voice_dataset import VoiceDataset
from training.tacotron2_model import Tacotron2
from training import DEFAULT_ALPHABET
from training.train import train, MINIMUM_MEMORY_GB, WEIGHT_DECAY
from training.validate import validate
from training.utils import (
    load_metadata,
    load_symbols,
    get_available_memory,
    get_learning_rate,
    get_batch_size,
    check_early_stopping,
    LEARNING_RATE_PER_64,
    BATCH_SIZE_PER_GB,
    BASE_SYMBOLS,
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

    def eval(self):
        pass

    def zero_grad(self):
        pass

    def state_dict(self):
        return self._state_dict


class MockedTacotron2Loss:
    class Loss:
        def item():
            return 0.5

    def __init__(self, *args):
        pass

    def __call__(self, *args):
        return self.Loss

    def item(self):
        return 0.5

    def backward(self):
        pass


class MockedOptimizer:
    param_groups = [{"lr": 0.1}]
    _state_dict = {"lr": 0.1}

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
@mock.patch("training.train.VoiceDataset", return_value=None)
@mock.patch("training.train.DataLoader", return_value=[(None, None), (None, None)])
@mock.patch("training.train.process_batch", return_value=((None, None), (None,)))
@mock.patch("torch.nn.utils.clip_grad_norm_")
@mock.patch("training.train.validate", return_value=(0.5, 0.5))
@mock.patch("training.train.calc_avgmax_attention", return_value=0.5)
def test_train(
    validate,
    clip_grad_norm_,
    process_batch,
    DataLoader,
    VoiceDataset,
    Adam,
    Tacotron2Loss,
    Tacotron2,
    get_available_memory,
    is_available,
    calc_avgmax_attention,
):
    metadata_path = os.path.join("test_samples", "dataset", "metadata.csv")
    dataset_directory = os.path.join("test_samples", "dataset", "wavs")
    output_directory = "checkpoint"
    train_size = 0.67

    train(
        metadata_path,
        dataset_directory,
        output_directory,
        epochs=1,
        batch_size=1,
        early_stopping=False,
        multi_gpu=False,
        train_size=train_size,
    )

    # Check checkpoint
    checkpoint_path = os.path.join(output_directory, "checkpoint_2")
    assert os.path.isfile(checkpoint_path)

    shutil.rmtree(output_directory)


# Validate
@mock.patch("training.validate.process_batch", return_value=((None,), (None,)))
@mock.patch("training.validate.calc_avgmax_attention", return_value=0.5)
def test_validate(process_batch, calc_avgmax_attention):
    loss, avgmax_attention = validate(MockedTacotron2(), [(None, None), (None, None)], MockedTacotron2Loss(), 0)
    assert loss == 0.5
    assert avgmax_attention == 0.5


# Clean text
def test_clean_text():
    text = clean_text("1st $500 Mr. 10.5 2,000 30 a\tb ~")
    assert text == "first five hundred dollars mister ten point five two thousand thirty a b "


# Dataset
@mock.patch("training.voice_dataset.clean_text", side_effect=lambda text: text)
def test_voice_dataset(clean_text):
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
    symbols = list("ABC")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model, optimizer, iteration, epoch = load_checkpoint(model_path, model, optimizer, [None] * 10000)

    assert iteration == 510000
    assert epoch == iteration // 10000

    checkpoint_folder = "test-checkpoints"
    os.makedirs(checkpoint_folder)
    save_checkpoint(model, optimizer, lr, iteration, symbols, epoch, checkpoint_folder, 1000, 1000)
    assert "checkpoint_510000" in os.listdir(checkpoint_folder)

    shutil.rmtree(checkpoint_folder)


class MockedEmbeddingLayer:
    weight = torch.zeros(3)


def test_transfer_symbols_embedding():
    original_embedding_weight = torch.Tensor([0.1, 0.2, 0.3])
    embedding_layer = MockedEmbeddingLayer()
    original_symbols = ["a", "c", "e"]
    new_symbols = ["a", "b", "é"]

    transfer_symbols_embedding(original_embedding_weight, embedding_layer, new_symbols, original_symbols)

    # Should match existing value in original_embedding_weight
    assert embedding_layer.weight[0] == 0.1
    # Should not match existing value in original_embedding_weight
    assert embedding_layer.weight[1] not in original_embedding_weight
    # Should map e -> é
    assert embedding_layer.weight[2] == 0.3


@mock.patch("os.remove")
def test_checkpoint_cleanup_should_remove(remove):
    # Old checkpoint (checkpoint_1000) should be removed
    checkpoint_cleanup("checkpoints", 2000, 1000, 10000)
    remove.assert_called_with(os.path.join("checkpoints", "checkpoint_1000"))


@mock.patch("os.remove")
def test_checkpoint_cleanup_should_not_remove(remove):
    # Backup checkpoint (checkpoint_20000) should not be removed
    checkpoint_cleanup("checkpoints", 21000, 1000, 10000)
    assert not remove.called


def test_warm_start_model():
    model_path = os.path.join("test_samples", "model.pt")
    model = Tacotron2()
    ignore_layers = ["embedding.weight"]
    symbols = list("ABC")
    model = warm_start_model(model_path, model, symbols, ignore_layers=ignore_layers)
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


# Memory
class FakeDeviceProperties:
    total_memory = 8 * 1024 * 1024 * 1024


@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.get_device_properties", return_value=FakeDeviceProperties)
@mock.patch("torch.cuda.memory_allocated", return_value=1 * 1024 * 1024 * 1024)
def test_get_available_memory(memory_allocated, get_device_properties, device_count):
    # 16GB Device memory - 2GB Usage
    assert get_available_memory() == 14


# Symbols
def test_load_symbols():
    alphabet_path = os.path.join("test_samples", "english.txt")
    symbols = set(load_symbols(alphabet_path))
    assert set(ascii_lowercase).issubset(symbols)
    assert set(BASE_SYMBOLS).issubset(symbols)


# Early stopping
def test_early_stopping():
    # Too few values
    assert check_early_stopping([10, 10, 10, 10]) is False

    # Loss still improving
    assert check_early_stopping([1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]) is False

    # Loss not improving
    assert check_early_stopping([0.5, 0.4999, 0.5, 0.4999, 0.5, 0.4998, 0.4999, 0.4996, 0.4997, 0.5]) is True


# Test parameters
def test_get_learning_rate():
    batch_size = 40
    lr = get_learning_rate(batch_size)
    assert lr == (batch_size / 64) ** 0.5 * LEARNING_RATE_PER_64


def test_get_batch_size():
    memory = 8
    batch_size = get_batch_size(memory)
    assert batch_size == int(memory * BATCH_SIZE_PER_GB)
