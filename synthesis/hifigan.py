import os
import json
import torch
from scipy.io.wavfile import write

from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

from synthesis.hifigan_model import Generator
from synthesis import Vocoder


MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    """
    Credit: https://github.com/jik876/hifi-gan
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Hifigan(Vocoder):
    def __init__(self, model_path, config_path):
        with open(config_path) as f:
            data = f.read()

        # Use GPU if available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        h = AttrDict(json.loads(data))
        self.model = Generator(h).to(device)

        checkpoint_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint_dict["generator"])
        self.model.eval()
        self.model.remove_weight_norm()

    def generate_audio(self, mel_output, path, sample_rate=22050):
        with torch.no_grad():
            if torch.cuda.is_available():
                mel_output = mel_output.type(torch.cuda.FloatTensor)

            y_g_hat = self.model(mel_output)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
            write(path, sample_rate, audio)
