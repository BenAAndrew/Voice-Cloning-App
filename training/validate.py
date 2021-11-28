import torch

from training.tacotron2_model.utils import process_batch
from training.utils import calc_avgmax_attention


def validate(model, val_loader, criterion, iteration):
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
    (float, float)
        Validation loss & Attention score
    """
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_avgmax_attention = 0
        for i, batch in enumerate(val_loader):
            y, y_pred = process_batch(batch, model)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            avgmax_attention = calc_avgmax_attention(batch[-1], batch[1], y_pred[-1])
            val_avgmax_attention += avgmax_attention
        val_loss = val_loss / (i + 1)
        val_avgmax_attention = val_avgmax_attention / (i + 1)

    model.train()
    return val_loss, val_avgmax_attention
