import os
import random
import argparse
import logging
from tqdm import tqdm

import sys

sys.path.append(os.path.abspath("../"))

logging.getLogger().setLevel(logging.INFO)

import torch
from torch.utils.data import DataLoader

from training.dataset import VoiceDataset
from training.tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss


def eval_checkpoint(
    metadata_path,
    dataset_directory,
    checkpoint_folder,
):
    # Hyperparams
    train_size = 0.8
    seed = 1234
    symbols = "_-!'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)

    # Load model
    logging.info("Loading model...")
    model = Tacotron2()
    criterion = Tacotron2Loss()
    logging.info("Loaded model")

    # Load data
    logging.info("Loading data...")
    with open(metadata_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split("|") for line in f]

    random.shuffle(filepaths_and_text)
    int(len(filepaths_and_text) * train_size)
    test_files = filepaths_and_text[-100:]
    valset = VoiceDataset(test_files, dataset_directory, symbols, seed)
    collate_fn = TextMelCollate()

    # Data loaders
    val_loader = DataLoader(valset, num_workers=1, sampler=None, batch_size=1, pin_memory=False, collate_fn=collate_fn)
    logging.info("Loaded data")

    for checkpoint in os.listdir(checkpoint_folder):
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint)
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint_dict["state_dict"])
        print("PROCESSING", checkpoint)

        model.train()
        model.zero_grad()
        losses = []
        for _, batch in enumerate(tqdm(val_loader)):
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            reduced_loss = loss.item()
            losses.append(reduced_loss)

        print("AVERAGE = ", sum(losses) / len(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metadata_path", type=str, help="metadata path")
    parser.add_argument("-d", "--dataset_directory", type=str, help="directory to dataset")
    parser.add_argument("-c", "--checkpoint_folder", type=str, help="checkpoint folder")

    args = parser.parse_args()
    eval_checkpoint(
        args.metadata_path,
        args.dataset_directory,
        args.checkpoint_folder,
    )
