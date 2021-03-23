import os
import random
import time
import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

import sys

sys.path.append("../")

import torch
from torch.utils.data import DataLoader

from training.dataset import VoiceDataset
from training.checkpoint import load_checkpoint, save_checkpoint, get_latest_checkpoint, warm_start_model
from training.validate import validate
from training.utils import get_available_memory, get_batch_size, get_learning_rate, check_space
from tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss


MINIMUM_MEMORY_GB = 4


def train(
    metadata_path,
    dataset_directory,
    output_directory,
    find_checkpoint=True,
    checkpoint_path=None,
    transfer_learning_path=None,
    epochs=8000,
    batch_size=None,
    logging=logging,
):
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

    # Hyperparams
    train_size = 0.8
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    iters_per_checkpoint = 1000
    seed = 1234
    symbols = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # Setup GPU
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # Load model & optimizer
    logging.info("Loading model...")
    model = Tacotron2().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = Tacotron2Loss()
    logging.info("Loaded model")

    # Load data
    logging.info("Loading data...")
    with open(metadata_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split("|") for line in f]

    random.shuffle(filepaths_and_text)
    train_cutoff = int(len(filepaths_and_text) * train_size)
    train_files = filepaths_and_text[:train_cutoff]
    test_files = filepaths_and_text[train_cutoff:]
    print(f"{len(train_files)} train files, {len(test_files)} test files")

    trainset = VoiceDataset(train_files, dataset_directory, symbols, seed)
    valset = VoiceDataset(test_files, dataset_directory, symbols, seed)
    collate_fn = TextMelCollate()

    # Data loaders
    train_loader = DataLoader(
        trainset, num_workers=1, sampler=None, batch_size=batch_size, pin_memory=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        valset, num_workers=1, sampler=None, batch_size=batch_size, pin_memory=False, collate_fn=collate_fn
    )
    logging.info("Loaded data")

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    if find_checkpoint and not checkpoint_path and not transfer_learning_path:
        checkpoint_path = get_latest_checkpoint(output_directory)

    if checkpoint_path:
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1
        epoch_offset = max(0, int(iteration / len(train_loader)))
        logging.info("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    elif transfer_learning_path:
        model = warm_start_model(transfer_learning_path, model)
        logging.info("Loaded transfer learning model '{}'".format(transfer_learning_path))

    num_iterations = len(train_loader) * epochs - epoch_offset
    check_space(num_iterations // iters_per_checkpoint)

    model.train()
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        logging.info(f"Progress - {epoch}/{epochs}")
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            reduced_loss = loss.item()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            optimizer.step()

            duration = time.perf_counter() - start
            logging.info(
                "Status - [Epoch {}: Iteration {}] Train loss {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    epoch, iteration, reduced_loss, grad_norm, duration
                )
            )

            if iteration % iters_per_checkpoint == 0:
                validate(model, val_loader, criterion, iteration)
                checkpoint_path = os.path.join(output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)
                logging.info(
                    "Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path)
                )

            iteration += 1

    validate(model, val_loader, criterion, iteration)
    checkpoint_path = os.path.join(output_directory, "checkpoint_{}".format(iteration))
    save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)
    logging.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
