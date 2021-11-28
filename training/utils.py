import torch
import random
import os
import unicodedata
from PIL import Image
from application.constants import(
    CHARACTER_ENCODING
)


from dataset import get_invalid_characters
from training.tacotron2_model.utils import get_mask_from_lengths


CHECKPOINT_SIZE_MB = 333
BATCH_SIZE_PER_GB = 2.5
LEARNING_RATE_PER_64 = 4e-4
MAXIMUM_LEARNING_RATE = 4e-4
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
    return min(
        (batch_size / 64) ** 0.5 * LEARNING_RATE_PER_64,  # Adam Learning Rate is proportional to sqrt(batch_size)
        MAXIMUM_LEARNING_RATE,
    )


def load_metadata(metadata_path):
    """
    Load metadata file and split entries into train and test.

    Parameters
    ----------
    metadata_path : str
        Path to metadata file

    Returns
    -------
    list
        List of samples
    """
    with open(metadata_path, encoding=CHARACTER_ENCODING) as f:
        filepaths_and_text = [line.strip().split("|") for line in f]
    random.shuffle(filepaths_and_text)
    return filepaths_and_text


def validate_dataset(filepaths_and_text, dataset_directory, symbols):
    missing_files = set()
    invalid_characters = set()
    wavs = os.listdir(dataset_directory)
    for filename, text in filepaths_and_text:
        text=text.lower()
        if filename not in wavs:
            missing_files.add(filename)
        invalid_characters_for_row = get_invalid_characters(text, symbols)
        if invalid_characters_for_row:
            invalid_characters.update(invalid_characters_for_row)

    assert not missing_files, f"Missing files: {(',').join(missing_files)}"
    detailed_info=""
    for c in invalid_characters:
        detailed_info+=f"{c} ({unicodedata.name(c)}),"
    assert not invalid_characters, f"Invalid characters in text (missing from language): {detailed_info}"


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
