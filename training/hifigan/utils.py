import os
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    return torch.load(filepath, map_location=device)


def get_checkpoint_options(checkpoint_directory):
    return list(set(sorted([int(filename.split("_")[1]) for filename in os.listdir(checkpoint_directory)], reverse=True)))


def save_checkpoints(generator, mpd, msd, optim_g, optim_d, iterations, epochs, output_directory, checkpoint_frequency, checkpoint_backup_frequency, logging):
    checkpoint_g_path = os.path.join(output_directory, f"g_{iterations}")
    checkpoint_do_path = os.path.join(output_directory, f"do_{iterations}")
    torch.save({"generator": (generator).state_dict()}, checkpoint_g_path)
    torch.save({
        "mpd": mpd.state_dict(),
        "msd": msd.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "steps": iterations,
        "epoch": epochs,
    }, checkpoint_do_path)
    logging.info(f"Saved checkpoints to {checkpoint_g_path} and {checkpoint_do_path}")

    # Remove last checkpoint if not a backup
    checkpoint_cleanup(output_directory, iterations, checkpoint_frequency, checkpoint_backup_frequency)


def checkpoint_cleanup(output_directory, iteration, checkpoint_frequency, checkpoint_backup_frequency):
    """
    Deletes previous checkpoint if it should be kept as a backup

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
