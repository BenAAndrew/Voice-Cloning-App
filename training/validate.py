import torch
from training.tacotron2_model.utils import parse_batch


def validate(model, val_loader, criterion, iteration, device):
    """
    Credit: https://github.com/NVIDIA/tacotron2

    Validate the tacotron2 model.

    Parameters
    ----------
    model : Tacotron2
        tacotron2 model
    val_loader : torch.utils.data.DataLoader
        Dataloader for the validation dataset
    criterion : func
        Loss function
    iteration : int
        Current training iteration

    Returns
    -------
    float
        Validation loss
    """
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y, mask_size, alignment_mask_size = parse_batch(batch)
            y_pred = model(x, mask_size=mask_size, alignment_mask_size=alignment_mask_size)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    return val_loss
