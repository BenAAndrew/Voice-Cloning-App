import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import itertools
import os

from meldataset import MelDataset, mel_spectrogram
from models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from utils import AttrDict, save_checkpoint, load_checkpoint

SEED = 1234
CONFIG_FILE = "config.json"
TRAIN_SIZE = 0.66


def train_hifigan(
    audio_folder,
    output_directory,
    checkpoint_g=None,
    checkpoint_do=None,
    epochs=2,
    batch_size=1,
    iters_per_checkpoint=10,
):
    assert torch.cuda.is_available(), "You do not have Torch with CUDA installed. Please check CUDA & Pytorch install"
    os.makedirs(output_directory, exist_ok=True)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda")
    audio = [os.path.join(audio_folder, name) for name in os.listdir(audio_folder)]
    train_cutoff = int(len(audio) * TRAIN_SIZE)
    train_files = audio[:train_cutoff]
    test_files = audio[train_cutoff:]

    with open(CONFIG_FILE) as f:
        params = AttrDict(json.loads(f.read()))

    generator = Generator(params).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if checkpoint_g and checkpoint_do:
        state_dict_g = load_checkpoint(checkpoint_g, device)
        state_dict_do = load_checkpoint(checkpoint_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        iterations = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
        print("last epoch", last_epoch)
    else:
        state_dict_do = None
        last_epoch = 0

    optim_g = torch.optim.AdamW(generator.parameters(), params.learning_rate, betas=[params.adam_b1, params.adam_b2])
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        params.learning_rate,
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
        n_cache_reuse=0,
        shuffle=True,
        fmax_loss=params.fmax_for_loss,
        device=device,
    )
    train_loader = DataLoader(
        trainset,
        num_workers=params.num_workers,
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
        False,
        False,
        n_cache_reuse=0,
        fmax_loss=params.fmax_for_loss,
        device=device,
    )
    validation_loader = DataLoader(
        validset, num_workers=1, shuffle=False, sampler=None, batch_size=batch_size, pin_memory=True, drop_last=True
    )

    generator.train()
    mpd.train()
    msd.train()
    iterations = 0
    for epoch in range(last_epoch, epochs):
        print("Epoch: {}".format(epoch + 1))

        for _, batch in enumerate(train_loader):
            print(len(batch))
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

            print(
                "Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}".format(
                    iterations, loss_gen_all, mel_error
                )
            )

            if iterations % iters_per_checkpoint == 0:
                # save checkpoint
                checkpoint_path = "{}/g_{:08d}".format(output_directory, iterations)
                save_checkpoint(checkpoint_path, {"generator": (generator).state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(output_directory, iterations)
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": mpd.state_dict(),
                        "msd": msd.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": iterations,
                        "epoch": epoch,
                    },
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
                    print("validation error", val_err)

                generator.train()

            iterations += 1

        scheduler_g.step()
        scheduler_d.step()

    checkpoint_path = "{}/g_{:08d}".format(output_directory, iterations)
    save_checkpoint(checkpoint_path, {"generator": (generator).state_dict()})
    checkpoint_path = "{}/do_{:08d}".format(output_directory, iterations)
    save_checkpoint(
        checkpoint_path,
        {
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "steps": iterations,
            "epoch": epoch,
        },
    )


if __name__ == "__main__":
    train_hifigan("wavs", "checkpoints", "checkpoints\\g_00000000", "checkpoints\\do_00000000")
