import os
import json
import torch
from scipy.io.wavfile import write

from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

from synthesis.hifigan_model import Generator


MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_hifigan_model(model_path, config_path):
    """
    Loads the Hifi-gan model.
    Uses GPU if available, otherwise uses CPU.

    Parameters
    ----------
    model_path : str
        Path to hifigan model
    config_path : str
        Path to hifigan config

    Returns
    -------
    Torch
        Loaded hifigan model
    """
    assert os.path.isfile(model_path)
    assert os.path.isfile(config_path)

    with open(config_path) as f:
        data = f.read()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    h = AttrDict(json.loads(data))
    generator = Generator(h).to(device)

    checkpoint_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()

    return generator


def generate_audio_hifigan(model, mel, filepath, sample_rate=22050):
    """
    Generates synthesised audio file.

    Parameters
    ----------
    model : Generator
        Hifigan model
    mel : list
        Synthesised mel data
    filepath : str
        Path to save generated audio to
    sample_rate : int (optional)
        Sample rate of audio (default is 22050)
    """
    with torch.no_grad():
        y_g_hat = model(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")
        write(filepath, sample_rate, audio)
