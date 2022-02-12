import os
from pathlib import Path
import random
from synthesis.synthesize import load_model
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

from training import DEFAULT_ALPHABET, SEED
from training.clean_text import clean_text
from training.voice_dataset import VoiceDataset
from training.checkpoint import load_checkpoint, save_checkpoint, warm_start_model
from training.validate import validate
from training.utils import (
    get_available_memory,
    get_batch_size,
    get_learning_rate,
    load_labels_file,
    check_early_stopping,
    calc_avgmax_attention,
    train_test_split,
    validate_dataset,
)
from training.tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss
from training.tacotron2_model.utils import process_batch
from synthesis.synthesize import text_to_sequence, generate_graph

MINIMUM_MEMORY_GB = 4
WEIGHT_DECAY = 1e-6
GRAD_CLIP_THRESH = 1.0
TRAINING_PATH = os.path.join("data", "training")


def train(
    audio_directory,
    output_directory,
    metadata_path=None,
    trainlist_path=None,
    vallist_path=None,
    symbols=DEFAULT_ALPHABET,
    checkpoint_path=None,
    transfer_learning_path=None,
    epochs=8000,
    batch_size=None,
    early_stopping=True,
    multi_gpu=True,
    iters_per_checkpoint=1000,
    iters_per_backup_checkpoint=10000,
    train_size=0.8,
    alignment_sentence="",
    logging=logging,
):
    """
    Trains the Tacotron2 model.

    Parameters
    ----------
    audio_directory : str
        Path to dataset clips
    output_directory : str
        Path to save checkpoints to
    metadata_path : str (optional)
        Path to label file
    trainlist_path : str (optional)
        Path to trainlist file
    vallist_path : str (optional)
        Path to vallist file
    symbols : list (optional)
        Valid symbols (default is English)
    checkpoint_path : str (optional)
        Path to a checkpoint to load (default is None)
    transfer_learning_path : str (optional)
        Path to a transfer learning checkpoint to use (default is None)
    epochs : int (optional)
        Number of epochs to run training for (default is 8000)
    batch_size : int (optional)
        Training batch size (calculated automatically if None)
    early_stopping : bool (optional)
        Whether to stop training when loss stops significantly decreasing (default is True)
    multi_gpu : bool (optional)
        Use multiple GPU's in parallel if available (default is True)
    iters_per_checkpoint : int (optional)
        How often temporary checkpoints are saved (number of iterations)
    iters_per_backup_checkpoint : int (optional)
        How often backup checkpoints are saved (number of iterations)
    train_size : float (optional)
        Percentage of samples to use for training (default is 80%/0.8)
    alignment_sentence : str (optional)
        Sentence for alignment graph to analyse performance
    logging : logging (optional)
        Logging object to write logs to

    Raises
    -------
    AssertionError
        If CUDA is not available or there is not enough GPU memory
    RuntimeError
        If the batch size is too high (causing CUDA out of memory)
    """
    assert metadata_path or (
        trainlist_path and vallist_path
    ), "You must give the path to your metadata file or trainlist/vallist files"
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
    if metadata_path:
        # metadata.csv
        filepaths_and_text = load_labels_file(metadata_path)
        random.shuffle(filepaths_and_text)
        train_files, test_files = train_test_split(filepaths_and_text, train_size)
    else:
        # trainlist.txt & vallist.txt
        train_files = load_labels_file(trainlist_path)
        test_files = load_labels_file(vallist_path)
        filepaths_and_text = train_files + test_files

    validate_dataset(filepaths_and_text, audio_directory, symbols)
    trainset = VoiceDataset(train_files, audio_directory, symbols)
    valset = VoiceDataset(test_files, audio_directory, symbols)
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

    if checkpoint_path:
        if transfer_learning_path:
            logging.info("Ignoring transfer learning as checkpoint already exists")
        model, optimizer, iteration, epoch_offset = load_checkpoint(checkpoint_path, model, optimizer, train_loader)
        iteration += 1
        logging.info("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    elif transfer_learning_path:
        model = warm_start_model(transfer_learning_path, model, symbols)
        logging.info("Loaded transfer learning model '{}'".format(transfer_learning_path))
    else:
        logging.info("Generating first checkpoint...")

    # Enable Multi GPU
    if multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Alignment sentence
    alignment_sequence = None
    alignment_folder = None
    if alignment_sentence:
        alignment_sequence = text_to_sequence(clean_text(alignment_sentence.strip(), symbols), symbols)
        alignment_folder = os.path.join(TRAINING_PATH, Path(output_directory).stem)
        os.makedirs(alignment_folder, exist_ok=True)

    model.train()
    validation_losses = []
    for epoch in range(epoch_offset, epochs):
        logging.info(f"Progress - {epoch}/{epochs}")
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            # Backpropogation
            model.zero_grad()
            y, y_pred = process_batch(batch, model)

            loss = criterion(y_pred, y)
            avgmax_attention = calc_avgmax_attention(batch[-1], batch[1], y_pred[-1])
            reduced_loss = loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
            optimizer.step()

            duration = time.perf_counter() - start
            logging.info(
                "Status - [Epoch {}: Iteration {}] Train loss {:.5f} Attention score {:.5f} {:.2f}s/it".format(
                    epoch, iteration, reduced_loss, avgmax_attention, duration
                )
            )

            # Validate & save checkpoint
            if iteration % iters_per_checkpoint == 0:
                logging.info("Validating model")
                val_loss, avgmax_attention = validate(model, val_loader, criterion, iteration)
                validation_losses.append(val_loss)
                logging.info(
                    "Saving model and optimizer state at iteration {} to {}. Validation score = {:.5f}, Attention score = {:.5f}".format(
                        iteration, output_directory, val_loss, avgmax_attention
                    )
                )
                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    learning_rate,
                    iteration,
                    symbols,
                    epoch,
                    output_directory,
                    iters_per_checkpoint,
                    iters_per_backup_checkpoint,
                )
                if alignment_sequence is not None:
                    try:
                        _, _, _, alignment = load_model(checkpoint_path).inference(alignment_sequence)
                        graph_path = os.path.join(alignment_folder, "checkpoint_{}.png".format(iteration))
                        generate_graph(alignment, graph_path, heading=f"Iteration {iteration}")
                        graph = os.path.relpath(graph_path).replace("\\", "/")
                        logging.info(f"Alignment - {iteration}, {graph}")
                    except Exception:
                        logging.info(
                            "Failed to generate alignment sample, you may need to train for longer before this is possible"
                        )

            iteration += 1

        # Early Stopping
        if early_stopping and check_early_stopping(validation_losses):
            logging.info("Stopping training early as loss is no longer decreasing")
            break

    logging.info(f"Progress - {epochs}/{epochs}")
    validate(model, val_loader, criterion, iteration)
    save_checkpoint(
        model,
        optimizer,
        learning_rate,
        iteration,
        symbols,
        epochs,
        output_directory,
        iters_per_checkpoint,
        iters_per_backup_checkpoint,
    )
    logging.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))


if __name__ == "__main__":
    """Train a tacotron2 model"""
    parser = argparse.ArgumentParser(description="Train a tacotron2 model")
    parser.add_argument("-m", "--metadata_path", type=str, help="metadata path")
    parser.add_argument("-a", "--audio_directory", type=str, help="directory to audio")
    parser.add_argument("-o", "--output_directory", type=str, help="directory to save checkpoints")
    parser.add_argument("-c", "--checkpoint_path", required=False, type=str, help="checkpoint path")
    parser.add_argument("-e", "--epochs", default=8000, type=int, help="num epochs")
    parser.add_argument("-b", "--batch_size", required=False, type=int, help="batch size")
    parser.add_argument(
        "-t",
        "--transfer_learning_path",
        required=False,
        type=str,
        help="path to an model to transfer learn from",
    )
    parser.add_argument("-a", "--alphabet_path", required=False, type=str, help="path to alphabet file")

    args = parser.parse_args()

    assert os.path.isfile(args.metadata_path)
    assert os.path.isdir(args.audio_directory)
    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)

    train(
        audio_directory=args.audio_directory,
        output_directory=args.output_directory,
        metadata_path=args.metadata_path,
        alphabet_path=args.alphabet_path,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        transfer_learning_path=args.transfer_learning_path,
    )
