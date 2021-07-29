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


def save_checkpoint(model, optimizer, learning_rate, iteration, epoch, output_directory, overwrite_checkpoints=True):
    """
    Save training checkpoint.
    Also deletes old checkpoints if overwrite_checkpoints is set.

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
    overwrite_checkpoints : bool (optional)
        Whether to delete old checkpoints in output_directory (default is True)
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

    if overwrite_checkpoints:
        for filename in os.listdir(output_directory):
            if filename != checkpoint_name:
                os.remove(os.path.join(output_directory, filename))
