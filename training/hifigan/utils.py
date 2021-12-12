import os
import torch


class AttrDict(dict):
    """
    Credit: https://github.com/jik876/hifi-gan
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    """
    Load a checkpoint

    Parameters
    ----------
    filepath : str
        Path to checkpoint
    device : torch.device
        GPU device

    Raises
    -------
    AssertionError
        If checkpoint file is not found

    Returns
    -------
    object
        Checkpoint object
    """
    assert os.path.isfile(filepath)
    return torch.load(filepath, map_location=device)


def get_checkpoint_options(checkpoint_directory):
    """
    List of available checkpoint iterations

    Parameters
    ----------
    checkpoint_directory : str
        Path to checkpoint folder

    Returns
    -------
    list
        List of checkpoint iterations sorted from high to low
    """
    return sorted(
        list(set([int(filename.split("_")[1]) for filename in os.listdir(checkpoint_directory)])), reverse=True
    )


def save_checkpoints(
    generator,
    mpd,
    msd,
    optim_g,
    optim_d,
    iterations,
    epochs,
    output_directory,
    checkpoint_frequency,
    checkpoint_backup_frequency,
    logging,
):
    """
    Save g_xxxx and do_xxxx checkpoints.
    Deletes old checkpoints that aren't backups

    Parameters
    ----------
    generator : Generator
        Hifigan model
    mpd : MultiPeriodDiscriminator
        MultiPeriodDiscriminator
    msd : MultiScaleDiscriminator
        MultiScaleDiscriminator
    optim_g : torch.optim.AdamW
        Generator optimiser
    optim_d : torch.optim.AdamW
        MultiPeriodDiscriminator & MultiScaleDiscriminator optimiser
    iterations : int
        Number of iterations
    epochs : int
        Number of epochs
    output_directory : str
        Folder to save checkpoints to
    checkpoint_frequency : int
        Frequency of checkpoint creation (in iterations)
    checkpoint_backup_frequency : int
        Frequency of checkpoint backups (in iterations)
    logging : logging
        Logging object to write logs to
    """
    checkpoint_g_path = os.path.join(output_directory, f"g_{iterations}")
    checkpoint_do_path = os.path.join(output_directory, f"do_{iterations}")
    torch.save({"generator": generator.state_dict()}, checkpoint_g_path)
    torch.save(
        {
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "steps": iterations,
            "epoch": epochs,
        },
        checkpoint_do_path,
    )
    logging.info(f"Saved checkpoints to {checkpoint_g_path} and {checkpoint_do_path}")

    # Remove last checkpoint if not a backup
    checkpoint_cleanup(output_directory, iterations, checkpoint_frequency, checkpoint_backup_frequency)


def checkpoint_cleanup(output_directory, iteration, checkpoint_frequency, checkpoint_backup_frequency):
    """
    Deletes previous checkpoint if it shouldn't be kept as a backup

    Parameters
    ----------
    output_directory : str
        Checkpoint folder
    iteration : int
        Current iteration
    checkpoint_frequency : int
        Frequency of checkpoint creation (in iterations)
    checkpoint_backup_frequency : int
        Frequency of checkpoint backups (in iterations)
    """
    if iteration > 0:
        last_checkpoint = iteration - checkpoint_frequency
        if last_checkpoint % checkpoint_backup_frequency != 0:
            # Last checkpoint shouldn't be kept as a backup
            try:
                os.remove(os.path.join(output_directory, f"g_{last_checkpoint}"))
                os.remove(os.path.join(output_directory, f"do_{last_checkpoint}"))
            except OSError:
                pass
