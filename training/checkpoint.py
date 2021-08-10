import os
import torch


def load_checkpoint(checkpoint_path, model, optimizer, train_loader):
    """
    Credit: https://github.com/NVIDIA/tacotron2

    Loads a given checkpoint to model & optimizer.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint
    model : Tacotron2
        tacotron2 model to load checkpoint into
    optimizer : torch.optim
        Torch optimizer
    train_loader: torch.Dataloader
        Torch training dataloader

    Returns
    -------
    Tacotron2
        Loaded tacotron2 model
    torch.optim
        Loaded optimizer
    int
        current iteration number
    """
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    iteration = checkpoint_dict["iteration"]
    epoch = checkpoint_dict.get("epoch", max(0, int(iteration / len(train_loader))))
    return model, optimizer, iteration, epoch


def warm_start_model(checkpoint_path, model, ignore_layers=["embedding.weight"]):
    """
    Credit: https://github.com/NVIDIA/tacotron2

    Warm start model for transfer learning.

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint
    model : Tacotron2
        tacotron2 model to load checkpoint into
    ignore_layers : list (optional)
        list of layers to ignore (default is ['embedding.weight'])

    Returns
    -------
    Tacotron2
        Loaded tacotron2 model
    """
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]
    if ignore_layers:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def get_state_dict(model):
    """
    Gets state dict for a given tacotron2 model.
    Handles parallel & non-parallel model types.

    Parameters
    ----------
    model : Tacotron2
        tacotron2 model

    Returns
    -------
    dict
        Model state dict
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()


def save_checkpoint(
    model,
    optimizer,
    learning_rate,
    iteration,
    epoch,
    output_directory,
    checkpoint_frequency,
    checkpoint_backup_frequency,
):
    """
    Save training checkpoint.
    Calls checkpoint cleanup on completion.

    Parameters
    ----------
    model : Tacotron2
        tacotron2 model
    optimizer : torch.optim
        Torch optimizer
    learning_rate : float
        Learning rate
    iteration : int
        Current iteration
    epoch : int
        Current epoch
    output_directory : str
        Folder to save checkpoint to
    checkpoint_frequency : int
        Frequency of checkpoint creation (in iterations)
    checkpoint_backup_frequency : int
        Frequency of checkpoint backups (in iterations)
    """
    checkpoint_name = "checkpoint_{}".format(iteration)
    torch.save(
        {
            "iteration": iteration,
            "state_dict": get_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "epoch": epoch,
        },
        os.path.join(output_directory, checkpoint_name),
    )
    checkpoint_cleanup(output_directory, iteration, checkpoint_frequency, checkpoint_backup_frequency)


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
                os.remove(os.path.join(output_directory, "checkpoint_{}".format(last_checkpoint)))
            except OSError:
                pass
