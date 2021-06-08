import os
import random
import time
import argparse
import logging
from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
logging.getLogger().setLevel(logging.INFO)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import VoiceDataset
from training.checkpoint import load_checkpoint, save_checkpoint, get_latest_checkpoint, warm_start_model
from training.validate import validate
from training.utils import get_available_memory, get_batch_size, get_learning_rate, check_space
from training.tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss
from training.tacotron2_model.utils import process_batch


MINIMUM_MEMORY_GB = 4
TRAIN_SIZE = 0.8
WEIGHT_DECAY = 1e-6
GRAD_CLIP_THRESH = 1.0
EARLY_STOPPING_WINDOW = 10
EARLY_STOPPING_MIN_DIFFERENCE = 0.0005
SEED = 1234
SYMBOLS = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def train(
    metadata_path,
    dataset_directory,
    output_directory,
    find_checkpoint=True,
    checkpoint_path=None,
    transfer_learning_path=None,
    overwrite_checkpoints=True,
    epochs=8000,
    batch_size=None,
    early_stopping=True,
    iters_per_checkpoint=1000,
    logging=logging,
):
    """
    Trains the Tacotron2 model.

    Parameters
    ----------
    metadata_path : str
        Path to label file
    dataset_directory : str
        Path to dataset clips
    output_directory : str
        Path to save checkpoints to
    find_checkpoint : bool (optional)
        Search for latest checkpoint to continue training from (default is True)
    checkpoint_path : str (optional)
        Path to a checkpoint to load (default is None)
    transfer_learning_path : str (optional)
        Path to a transfer learning checkpoint to use (default is None)
    overwrite_checkpoints : bool (optional)
        Whether to overwrite old checkpoints (default is True)
    epochs : int (optional)
        Number of epochs to run training for (default is 8000)
    batch_size : int (optional)
        Training batch size (calculated automatically if None)
    early_stopping : bool (optional)
        Whether to stop training when loss stops significantly decreasing (default is True)
    iters_per_checkpoint : int (optional)
        How often checkpoints are saved (number of iterations)
    logging : logging (optional)
        Logging object to write logs to

    Raises
    -------
    AssertionError
        If CUDA is not available or there is not enough GPU memory
    RuntimeError
        If the batch size is too high (causing CUDA out of memory)
    """
    assert torch.cuda.is_available(), "You do not have Torch with CUDA installed. Please check CUDA & Pytorch install"
    os.makedirs(output_directory, exist_ok=True)

    available_memory_gb = get_available_memory()
    assert (
        available_memory_gb >= MINIMUM_MEMORY_GB
    ), f"Required GPU with at least {MINIMUM_MEMORY_GB}GB memory. (only {available_memory_gb}GB available)"

    if not batch_size:
        batch_size = get_batch_size(available_memory_gb)

    learning_rate = get_learning_rate(batch_size)
    logging.info(
        f"Setting batch size to {batch_size}, learning rate to {learning_rate}. ({available_memory_gb}GB GPU memory free)"
    )

    # Set seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)

    # Setup GPU
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # Load model & optimizer
    logging.info("Loading model...")
    model = Tacotron2().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    criterion = Tacotron2Loss()
    logging.info("Loaded model")

    # Load data
    logging.info("Loading data...")
    with open(metadata_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split("|") for line in f]

    random.shuffle(filepaths_and_text)
    train_cutoff = int(len(filepaths_and_text) * TRAIN_SIZE)
    train_files = filepaths_and_text[:train_cutoff]
    test_files = filepaths_and_text[train_cutoff:]
    print(f"{len(train_files)} train files, {len(test_files)} test files")

    trainset = VoiceDataset(train_files, dataset_directory, SYMBOLS, SEED)
    valset = VoiceDataset(test_files, dataset_directory, SYMBOLS, SEED)
    collate_fn = TextMelCollate()

    # Data loaders
    train_loader = DataLoader(
        trainset, num_workers=0, sampler=None, batch_size=batch_size, pin_memory=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valset, num_workers=0, sampler=None, batch_size=batch_size, pin_memory=False, collate_fn=collate_fn
    )
    logging.info("Loaded data")

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    if find_checkpoint and not checkpoint_path:
        checkpoint_path = get_latest_checkpoint(output_directory)

    if checkpoint_path:
        if transfer_learning_path:
            logging.info("Ignoring transfer learning as checkpoint already exists")
        model, optimizer, iteration, epoch_offset = load_checkpoint(checkpoint_path, model, optimizer, train_loader)
        iteration += 1
        logging.info("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    elif transfer_learning_path:
        model = warm_start_model(transfer_learning_path, model)
        logging.info("Loaded transfer learning model '{}'".format(transfer_learning_path))

    # Check available memory
    if not overwrite_checkpoints:
        num_iterations = len(train_loader) * epochs - epoch_offset
        check_space(num_iterations // iters_per_checkpoint)

    # Enable Multi GPU
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.train()
    validation_losses = []
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        logging.info(f"Progress - {epoch}/{epochs}")
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            # Backpropogation
            model.zero_grad()
            y, y_pred = process_batch(batch, model)

            loss = criterion(y_pred, y)
            reduced_loss = loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
            optimizer.step()

            duration = time.perf_counter() - start
            logging.info(
                "Status - [Epoch {}: Iteration {}] Train loss {:.6f} {:.2f}s/it".format(
                    epoch, iteration, reduced_loss, duration
                )
            )

            # Validate & save checkpoint
            if iteration % iters_per_checkpoint == 0:
                val_loss = validate(model, val_loader, criterion, iteration)
                validation_losses.append(val_loss)
                logging.info(
                    "Saving model and optimizer state at iteration {} to {}. Scored {}".format(
                        iteration, output_directory, val_loss
                    )
                )
                save_checkpoint(
                    model, optimizer, learning_rate, iteration, epoch, output_directory, overwrite_checkpoints
                )

            iteration += 1

        # Early Stopping
        if early_stopping and len(validation_losses) >= EARLY_STOPPING_WINDOW:
            losses = validation_losses[-EARLY_STOPPING_WINDOW:]
            difference = max(losses) - min(losses)
            if difference < EARLY_STOPPING_MIN_DIFFERENCE:
                logging.info("Stopping training early as loss is no longer decreasing")
                break

    logging.info(f"Progress - {epochs}/{epochs}")
    validate(model, val_loader, criterion, iteration)
    save_checkpoint(model, optimizer, learning_rate, iteration, epochs, output_directory, overwrite_checkpoints)
    logging.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))


if __name__ == "__main__":
    """Train a tacotron2 model"""
    parser = argparse.ArgumentParser(description="Train a tacotron2 model")
    parser.add_argument("-m", "--metadata_path", type=str, help="metadata path")
    parser.add_argument("-d", "--dataset_directory", type=str, help="directory to dataset")
    parser.add_argument("-o", "--output_directory", type=str, help="directory to save checkpoints")
    parser.add_argument("-l", "--find_checkpoint", default=True, type=str, help="load checkpoint if one exists")
    parser.add_argument("-c", "--checkpoint_path", required=False, type=str, help="checkpoint path")
    parser.add_argument("-e", "--epochs", default=8000, type=int, help="num epochs")
    parser.add_argument("-b", "--batch_size", required=False, type=int, help="batch size")

    args = parser.parse_args()

    assert os.path.isfile(args.metadata_path)
    assert os.path.isdir(args.dataset_directory)
    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)

    train(
        args.metadata_path,
        args.dataset_directory,
        args.output_directory,
        args.find_checkpoint,
        args.checkpoint_path,
        args.epochs,
        args.batch_size,
    )
