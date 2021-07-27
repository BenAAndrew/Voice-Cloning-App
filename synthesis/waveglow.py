import torch
import IPython.display as ipd

from synthesis import Vocoder


class Waveglow(Vocoder):
    def __init__(self, path):
        self.waveglow = torch.load(path)["model"]
        # Use GPU if available
        if torch.cuda.is_available():
            self.waveglow.cuda().eval().half()
        for k in self.waveglow.convinv:
            k.float()

    def generate_audio(self, mel_output, path, sample_rate=22050):
        with torch.no_grad():
            audio = self.waveglow.infer(mel_output, sigma=0.666)

        audio = audio[0].data.cpu().numpy()
        audio = ipd.Audio(audio, rate=sample_rate)
        with open(path, "wb") as f:
            f.write(audio.data)
