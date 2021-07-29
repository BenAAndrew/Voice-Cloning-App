from abc import ABC, abstractmethod


MAX_WAV_VALUE = 32768.0


class Vocoder(ABC):
    """
    Produces audio data for tacotron2 mel spectrogram output
    """

    @abstractmethod
    def generate_audio(self, mel_output):
        """
        Produces wav audio data for a given mel output.

        Parameters
        ----------
        mel_output : Tensor
            Mel spectrogram output

        Returns
        -------
        np.array
            Generated audio data
        """
        pass
