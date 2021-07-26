import os
import inflect
import random
from string import ascii_lowercase
from unittest import mock

from dataset.clip_generator import CHARACTER_ENCODING
from training.clean_text import clean_text
from training.dataset import VoiceDataset
from training.train import DEFAULT_ALPHABET
from training.utils import check_space, load_symbols, CHECKPOINT_SIZE_MB, PUNCTUATION


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
