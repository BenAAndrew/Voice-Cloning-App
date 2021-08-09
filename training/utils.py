import shutil
import torch
import random


from dataset.clip_generator import CHARACTER_ENCODING


CHECKPOINT_SIZE_MB = 333
BATCH_SIZE_PER_GB = 2.5
LEARNING_RATE_PER_BATCH = 3.125e-5
PUNCTUATION = list("_-!'(),.:;? ")
EARLY_STOPPING_WINDOW = 10
EARLY_STOPPING_MIN_DIFFERENCE = 0.0005


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
        gpu_memory = torch.cuda.get_device_properties(i).total_memory
        memory_in_use = torch.cuda.memory_allocated(i)
        available_memory = gpu_memory - memory_in_use
        available_memory_gb += available_memory // 1024 // 1024 // 1024

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
    return batch_size * LEARNING_RATE_PER_BATCH


def load_metadata(metadata_path, train_size):
    """
    Load metadata file and split entries into train and test.

    Parameters
    ----------
    metadata_path : str
        Path to metadata file
    train_size : float
        Percentage of entries to use for training (rest used for testing)

    Returns
    -------
    (list, list)
        List of train and test samples
    """
    with open(metadata_path, encoding=CHARACTER_ENCODING) as f:
        filepaths_and_text = [line.strip().split("|") for line in f]

    random.shuffle(filepaths_and_text)
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
    symbols = PUNCTUATION.copy()

    with open(alphabet_file) as f:
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
