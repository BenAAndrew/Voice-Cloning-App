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

from training.voice_dataset import VoiceDataset
from training.checkpoint import load_checkpoint, save_checkpoint, warm_start_model
from training.validate import validate
from training.utils import (
    get_available_memory,
    get_batch_size,
    get_learning_rate,
    load_metadata,
    load_symbols,
    check_early_stopping,
)
from training.tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss
from training.tacotron2_model.utils import process_batch, get_mask_from_lengths

MINIMUM_MEMORY_GB = 4
WEIGHT_DECAY = 1e-6
GRAD_CLIP_THRESH = 1.0
SEED = 1234
DEFAULT_ALPHABET = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def train(
    metadata_path,
    dataset_directory,
    output_directory,
    alphabet_path=None,
    checkpoint_path=None,
    transfer_learning_path=None,
    epochs=8000,
    batch_size=None,
    early_stopping=True,
    multi_gpu=True,
    iters_per_checkpoint=1000,
    iters_per_backup_checkpoint=10000,
    train_size=0.8,
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
    alphabet_path : str
        Path to alphabet file (default is English)
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
    train_files, test_files = load_metadata(metadata_path, train_size)
    symbols = load_symbols(alphabet_path) if alphabet_path else DEFAULT_ALPHABET
    trainset = VoiceDataset(train_files, dataset_directory, symbols)
    valset = VoiceDataset(test_files, dataset_directory, symbols)
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
            avg_prob = calc_avgmax_attention(batch[-1], batch[1], y_pred[-1])
            reduced_loss = loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
            optimizer.step()

            duration = time.perf_counter() - start
            logging.info(
                "Status - [Epoch {}: Iteration {}] Train loss {:.6f} {:.2f}s/it {:.2f}att_str".format(
                    epoch, iteration, reduced_loss, duration, avg_prob
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
    attention_mask = txt_mask.unsqueeze(1) & mel_mask.unsqueeze(2)# [B, mel_T, 1] * [B, 1, txt_T] -> [B, mel_T, txt_T]
    
    alignment = alignment.data.masked_fill(~attention_mask, 0.0)
    avg_prob = alignment.data.amax(dim=2).sum(1).div(mel_lengths.to(alignment)).mean().item() # [B, mel_T, txt_T]
    return avg_prob


if __name__ == "__main__":
    """Train a tacotron2 model"""
    parser = argparse.ArgumentParser(description="Train a tacotron2 model")
    parser.add_argument("-m", "--metadata_path", type=str, help="metadata path")
    parser.add_argument("-d", "--dataset_directory", type=str, help="directory to dataset")
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

    args = parser.parse_args()

    assert os.path.isfile(args.metadata_path)
    assert os.path.isdir(args.dataset_directory)
    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)

    train(
        metadata_path=args.metadata_path,
        dataset_directory=args.dataset_directory,
        output_directory=args.output_directory,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        transfer_learning_path=args.transfer_learning_path,
    )
