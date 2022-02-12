import torch
import random
import os
import unicodedata
from PIL import Image

from dataset import CHARACTER_ENCODING
from dataset.utils import get_invalid_characters
from training import BASE_SYMBOLS, SEED, TRAIN_FILE, VALIDATION_FILE
from training.tacotron2_model.utils import get_mask_from_lengths
from training.clean_text import clean_text

CHECKPOINT_SIZE_MB = 333
BATCH_SIZE_PER_GB = 2.5
LEARNING_RATE_PER_64 = 4e-4
MAXIMUM_LEARNING_RATE = 4e-4
EARLY_STOPPING_WINDOW = 10
EARLY_STOPPING_MIN_DIFFERENCE = 0.0005


def get_gpu_memory(gpu_index):
    """
    Get available memory of a GPU.

    Parameters
    ----------
    gpu_index : int
        Index of GPU

    Returns
    -------
    int
        Available GPU memory in GB
    """
    gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory
    memory_in_use = torch.cuda.memory_allocated(gpu_index)
    available_memory = gpu_memory - memory_in_use
    return available_memory // 1024 // 1024 // 1024


def get_available_memory():
    """
    Get available GPU memory in GB.

    Returns
    -------
    int
        Available GPU memory in GB
    """
    available_memory_gb = 0

    for i in range(torch.cuda.device_count()):
        available_memory_gb += get_gpu_memory(i)

    return available_memory_gb


def get_batch_size(available_memory_gb):
    """
    Calulate batch size.

    Parameters
    ----------
    available_memory_gb : int
        Available GPU memory in GB

    Returns
    -------
    int
        Batch size
    """
    return int(available_memory_gb * BATCH_SIZE_PER_GB)


def get_learning_rate(batch_size):
    """
    Calulate learning rate.

    Parameters
    ----------
    batch_size : int
        Batch size

    Returns
    -------
    float
        Learning rate
    """
    return min(
        (batch_size / 64) ** 0.5 * LEARNING_RATE_PER_64,  # Adam Learning Rate is proportional to sqrt(batch_size)
        MAXIMUM_LEARNING_RATE,
    )


def load_labels_file(filepath):
    """
    Load labels file

    Parameters
    ----------
    filepath : str
        Path to text file

    Returns
    -------
    list
        List of samples
    """
    with open(filepath, encoding=CHARACTER_ENCODING) as f:
        return [line.strip().split("|") for line in f]


def validate_dataset(filepaths_and_text, dataset_directory, symbols):
    """
    Validates dataset has required files and a valid character set

    Parameters
    ----------
    filepaths_and_text : list
        List of samples
    dataset_directory : str
        Path to dataset audio directory
    symbols : list
        List of supported symbols

    Raises
    -------
    AssertionError
        If files are missing or invalid characters are found
    """
    missing_files = set()
    invalid_characters = set()
    wavs = os.listdir(dataset_directory)
    for filename, text in filepaths_and_text:
        text = clean_text(text, remove_invalid_characters=False)
        if filename not in wavs:
            missing_files.add(filename)
        invalid_characters_for_row = get_invalid_characters(text, symbols)
        if invalid_characters_for_row:
            invalid_characters.update(invalid_characters_for_row)

    assert not missing_files, f"Missing files: {(',').join(missing_files)}"
    assert (
        not invalid_characters
    ), f"Invalid characters in text (for alphabet): {','.join([f'{c} ({unicodedata.name(c)})' for c in invalid_characters])}"


def train_test_split(filepaths_and_text, train_size):
    """
    Split dataset into train & test data

    Parameters
    ----------
    filepaths_and_text : list
        List of samples
    train_size : float
        Percentage of entries to use for training (rest used for testing)

    Returns
    -------
    (list, list)
        List of train and test samples
    """
    train_cutoff = int(len(filepaths_and_text) * train_size)
    train_files = filepaths_and_text[:train_cutoff]
    test_files = filepaths_and_text[train_cutoff:]
    print(f"{len(train_files)} train files, {len(test_files)} test files")
    return train_files, test_files


def load_symbols(alphabet_file):
    """
    Get alphabet and punctuation for a given alphabet file.

    Parameters
    ----------
    alphabet_file : str
        Path to alphabnet file

    Returns
    -------
    list
        List of symbols (punctuation + alphabet)
    """
    symbols = BASE_SYMBOLS.copy()

    with open(alphabet_file, encoding=CHARACTER_ENCODING) as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]

    for line in lines:
        if line not in symbols:
            symbols.append(line)

    return symbols


def check_early_stopping(validation_losses):
    """
    Decide whether to stop training depending on validation losses.

    Parameters
    ----------
    validation_losses : list
        List of validation loss scores

    Returns
    -------
    bool
        True if training should stop, otherwise False
    """
    if len(validation_losses) >= EARLY_STOPPING_WINDOW:
        losses = validation_losses[-EARLY_STOPPING_WINDOW:]
        difference = max(losses) - min(losses)
        if difference < EARLY_STOPPING_MIN_DIFFERENCE:
            return True
    return False


def calc_avgmax_attention(mel_lengths, text_lengths, alignment):
    """
    Calculate Average Max Attention for Tacotron2 Alignment.
    Roughly represents how well the model is linking the text to the audio.
    Low values during training typically result in unstable speech during inference.

    Parameters
    ----------
    mel_lengths : torch.Tensor
        lengths of each mel in the batch
    text_lengths : torch.Tensor
        lengths of each text in the batch
    alignment : torch.Tensor
        alignments from model of shape [B, mel_length, text_length]

    Returns
    -------
    float
        average max attention
    """
    mel_mask = get_mask_from_lengths(mel_lengths, device=alignment.device)
    txt_mask = get_mask_from_lengths(text_lengths, device=alignment.device)
    # [B, mel_T, 1] * [B, 1, txt_T] -> [B, mel_T, txt_T]
    attention_mask = txt_mask.unsqueeze(1) & mel_mask.unsqueeze(2)

    alignment = alignment.data.masked_fill(~attention_mask, 0.0)
    # [B, mel_T, txt_T]
    avg_prob = alignment.data.amax(dim=2).sum(1).div(mel_lengths.to(alignment)).mean().item()
    return avg_prob


def generate_timelapse_gif(folder, output_path):
    """
    Generates a GIF timelapse from a folder of images.

    Parameters
    ----------
    folder : str
        Path to folder of images
    output_path : str
        Path to save resulting GIF to
    """
    images = sorted(os.listdir(folder), key=lambda filename: int(filename.split("_")[1].split(".")[0]))
    frames = [Image.open(os.path.join(folder, image)) for image in images]
    frames[0].save(output_path, format="GIF", append_images=frames[1:], save_all=True, duration=200, loop=0)


def create_trainlist_vallist_files(folder, metadata_path, train_size=0.8):
    """
    Creates trainlist & vallist files for compatibility with other notebooks.

    Parameters
    ----------
    folder : str
        Destination folder
    metadata_path : str
        Path to metadata file
    train_size : float (optional)
        Percentage of samples to use for training (default is 80%/0.8)
    """
    random.seed(SEED)
    filepaths_and_text = load_labels_file(metadata_path)
    random.shuffle(filepaths_and_text)
    train_files, test_files = train_test_split(filepaths_and_text, train_size)

    with open(os.path.join(folder, TRAIN_FILE), "w") as f:
        for line in train_files:
            f.write(f"{line[0]}|{line[1]}\n")

    with open(os.path.join(folder, VALIDATION_FILE), "w") as f:
        for line in test_files:
            f.write(f"{line[0]}|{line[1]}\n")
