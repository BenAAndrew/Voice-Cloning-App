from abc import ABC, abstractmethod


class Vocoder(ABC):
    @abstractmethod
    def generate_audio(self, mel_output, path, sample_rate):
        pass
