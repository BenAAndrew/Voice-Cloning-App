import os
import torch


def get_latest_checkpoint(checkpoint_folder):
    checkpoints = os.listdir(checkpoint_folder)
    if not checkpoints:
        return None

    latest_checkpoint = checkpoints[0]
    if len(checkpoints) > 1:
        for checkpoint in checkpoints:
            if int(checkpoint.split("_")[1].split(".")[0]) > int(latest_checkpoint.split("_")[1].split(".")[0]):
                latest_checkpoint = checkpoint

    return os.path.join(checkpoint_folder, latest_checkpoint)


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    iteration = checkpoint_dict["iteration"]
    return model, optimizer, iteration


def warm_start_model(checkpoint_path, model, ignore_layers=["embedding.weight"]):
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]
    if ignore_layers:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def save_checkpoint(model, optimizer, learning_rate, iteration, output_directory, overwrite_checkpoints=True):
    checkpoint_name = "checkpoint_{}".format(iteration)
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        os.path.join(output_directory, checkpoint_name),
    )

    if overwrite_checkpoints:
        for filename in os.listdir(output_directory):
            if filename != checkpoint_name:
                os.remove(os.path.join(output_directory, filename))
