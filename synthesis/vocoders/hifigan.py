import json
import torch

from synthesis.vocoders.hifigan_model import Generator
from synthesis.vocoders.vocoder import Vocoder, MAX_WAV_VALUE


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

    def generate_audio(self, mel_output):
        with torch.no_grad():
            if torch.cuda.is_available():
                mel_output = mel_output.type(torch.cuda.FloatTensor)

            y_g_hat = self.model(mel_output)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")
            return audio
