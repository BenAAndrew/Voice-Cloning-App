from abc import ABC, abstractmethod


MAX_WAV_VALUE = 32768.0


class Vocoder(ABC):
    @abstractmethod
    def generate_audio(self, mel_output):
        pass
