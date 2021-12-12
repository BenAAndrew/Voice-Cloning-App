import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import itertools
import logging
import json
import time
import os
from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
logging.getLogger().setLevel(logging.INFO)

from training.hifigan.meldataset import MelDataset, mel_spectrogram
from training.hifigan.models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from training.hifigan.utils import AttrDict, save_checkpoints, load_checkpoint

from training.utils import get_gpu_memory


SEED = 1234
CONFIG_FILE = os.path.join(dirname(abspath(__file__)), "config.json")
BATCH_SIZE_PER_GB = 0.8
LEARNING_RATE_PER_64 = 8e-4


def train(
    audio_folder,
    output_directory,
    checkpoint_g=None,
    checkpoint_do=None,
    epochs=1000,
    batch_size=None,
    iters_per_checkpoint=1000,
    iters_per_backup_checkpoint=10000,
    train_size=0.8,
    logging=logging,
):
    """
    Credit: https://github.com/jik876/hifi-gan

    Trains the Hifigan model.

    Parameters
    ----------
    audio_folder : str
        Path to audio folder
    output_directory : str
        Path to save checkpoints to
    checkpoint_g : str (optional)
        Path to g checkpoint
    checkpoint_do : str (optional)
        Path to do checkpoint
    epochs : int (optional)
        Number of epochs to run training for
    batch_size : int (optional)
        Training batch size (calculated automatically if None)
    iters_per_checkpoint : int (optional)
        How often temporary checkpoints are saved (number of iterations)
    iters_per_backup_checkpoint : int (optional)
        How often backup checkpoints are saved (number of iterations)
    train_size : float (optional)
        Percentage of samples to use for training
    logging : logging (optional)
        Logging object to write logs to

    Raises
    -------
    AssertionError
        If CUDA is not available
    RuntimeError
        If the batch size is too high (causing CUDA out of memory)
    """
    assert torch.cuda.is_available(), "You do not have Torch with CUDA installed. Please check CUDA & Pytorch install"
    os.makedirs(output_directory, exist_ok=True)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device("cuda")

    audio = [os.path.join(audio_folder, name) for name in os.listdir(audio_folder)]
    train_cutoff = int(len(audio) * train_size)
    train_files = audio[:train_cutoff]
    test_files = audio[train_cutoff:]
    print(f"{len(train_files)} train files, {len(test_files)} test files")

    available_memory = get_gpu_memory(0)
    if not batch_size:
        batch_size = int(available_memory * BATCH_SIZE_PER_GB)
    learning_rate = (
        batch_size / 64
    ) ** 0.5 * LEARNING_RATE_PER_64  # Adam Learning Rate is proportional to sqrt(batch_size)
    logging.info(
        f"Setting batch size to {batch_size}, learning rate to {learning_rate}. ({available_memory}GB GPU memory free)"
    )

    with open(CONFIG_FILE) as f:
        params = AttrDict(json.loads(f.read()))

    generator = Generator(params).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if checkpoint_g and checkpoint_do:
        logging.info(f"Loading {checkpoint_g} and {checkpoint_do} checkpoints")
        state_dict_g = load_checkpoint(checkpoint_g, device)
        state_dict_do = load_checkpoint(checkpoint_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        iterations = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
    else:
        state_dict_do = None
        iterations = 0
        last_epoch = -1

    optim_g = torch.optim.AdamW(generator.parameters(), learning_rate, betas=[params.adam_b1, params.adam_b2])
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        learning_rate,
        betas=[params.adam_b1, params.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=params.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=params.lr_decay, last_epoch=last_epoch)

    trainset = MelDataset(
        train_files,
        params.segment_size,
        params.n_fft,
        params.num_mels,
        params.hop_size,
        params.win_size,
        params.sampling_rate,
        params.fmin,
        params.fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=0,
        fmax_loss=params.fmax_for_loss,
        device=device,
    )
    train_loader = DataLoader(
        trainset,
        num_workers=0,
        shuffle=False,
        sampler=None,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )

    validset = MelDataset(
        test_files,
        params.segment_size,
        params.n_fft,
        params.num_mels,
        params.hop_size,
        params.win_size,
        params.sampling_rate,
        params.fmin,
        params.fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=0,
        fmax_loss=params.fmax_for_loss,
        device=device,
    )
    validation_loader = DataLoader(
        validset, num_workers=0, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True
    )

    generator.train()
    mpd.train()
    msd.train()
    last_epoch = min(last_epoch, 0)
    for epoch in range(last_epoch, epochs):
        print("Epoch: {}".format(epoch + 1))

        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1),
                params.n_fft,
                params.num_mels,
                params.sampling_rate,
                params.hop_size,
                params.win_size,
                params.fmin,
                params.fmax_for_loss,
            )

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            with torch.no_grad():
                mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

            duration = time.perf_counter() - start
            logging.info(
                "Status - [Epoch {}: Iteration {}] Loss {:4.3f} Mel-Spec. Error {:.5f} {:.2f}s/it".format(
                    epoch, iterations, loss_gen_all, mel_error, duration
                )
            )

            if iterations % iters_per_checkpoint == 0 and iterations != 0:
                save_checkpoints(
                    generator,
                    mpd,
                    msd,
                    optim_g,
                    optim_d,
                    iterations,
                    epochs,
                    output_directory,
                    iters_per_checkpoint,
                    iters_per_backup_checkpoint,
                    logging,
                )

                # validate
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, y, _, y_mel = batch
                        y_g_hat = generator(x.to(device))
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                        y_g_hat_mel = mel_spectrogram(
                            y_g_hat.squeeze(1),
                            params.n_fft,
                            params.num_mels,
                            params.sampling_rate,
                            params.hop_size,
                            params.win_size,
                            params.fmin,
                            params.fmax_for_loss,
                        )
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                    val_err = val_err_tot / (j + 1)
                    logging.info(f"Validation error: {val_err}")

                generator.train()

            iterations += 1

        scheduler_g.step()
        scheduler_d.step()

    save_checkpoints(
        generator,
        mpd,
        msd,
        optim_g,
        optim_d,
        iterations,
        epochs,
        output_directory,
        iters_per_checkpoint,
        iters_per_backup_checkpoint,
        logging,
    )


if __name__ == "__main__":
    """Train a tacotron2 model"""
    parser = argparse.ArgumentParser(description="Train a tacotron2 model")
    parser.add_argument("-d", "--dataset_directory", required=True, type=str, help="directory to dataset")
    parser.add_argument("-o", "--output_directory", required=True, type=str, help="directory to save checkpoints")
    parser.add_argument("--generator_checkpoint_path", required=False, type=str, help="generator checkpoint path")
    parser.add_argument(
        "--discriminator_checkpoint_path", required=False, type=str, help="discriminator checkpoint path"
    )
    parser.add_argument("-e", "--epochs", default=1000, type=int, help="num epochs")
    parser.add_argument("-b", "--batch_size", required=False, type=int, help="batch size")
    parser.add_argument("-i", "--iters_per_checkpoint", default=1000, type=int, help="iters per checkpoint")
    args = parser.parse_args()

    train(
        audio_folder=args.dataset_directory,
        output_directory=args.output_directory,
        checkpoint_g=args.generator_checkpoint_path,
        checkpoint_do=args.discriminator_checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        iters_per_checkpoint=args.iters_per_checkpoint,
    )
