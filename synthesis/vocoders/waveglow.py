import torch

from synthesis.vocoders.vocoder import Vocoder, MAX_WAV_VALUE


class Waveglow(Vocoder):
    def __init__(self, path):
        self.waveglow = torch.load(path)["model"]
        # Use GPU if available
        if torch.cuda.is_available():
            self.waveglow.cuda().eval().half()
        for k in self.waveglow.convinv:
            k.float()

    def generate_audio(self, mel_output):
        with torch.no_grad():
            audio = self.waveglow.infer(mel_output, sigma=0.666)

        audio = audio.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")
        return audio
