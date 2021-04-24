import torch
import IPython.display as ipd


def load_waveglow_model(waveglow_path):
    """
    Loads the Waveglow model.
    Uses GPU if available, otherwise uses CPU.

    Parameters
    ----------
    waveglow_path : str
        Path to waveglow model

    Returns
    -------
    Torch
        Loaded waveglow model
    """
    waveglow = torch.load(waveglow_path)["model"]
    if torch.cuda.is_available():
        waveglow.cuda().eval().half()

    for k in waveglow.convinv:
        k.float()
    return waveglow


def generate_audio_waveglow(model, mel, filepath, sample_rate=22050):
    """
    Generates synthesised audio file.

    Parameters
    ----------
    model : Torch
        Waveglow model
    mel : list
        Synthesised mel data
    filepath : str
        Path to save generated audio to
    sample_rate : int (optional)
        Sample rate of audio (default is 22050)
    """
    with torch.no_grad():
        audio = model.infer(mel, sigma=0.666)

    audio = audio[0].data.cpu().numpy()
    audio = ipd.Audio(audio, rate=sample_rate)
    with open(filepath, "wb") as f:
        f.write(audio.data)
