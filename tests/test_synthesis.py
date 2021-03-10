import os
import inflect

from synthesis.synthesize import load_model, load_waveglow, synthesize
from dataset.transcribe import transcribe
from tests.utils import TestClass, text_similarity


class TestSynthesis(TestClass):
    def setup_class(self):
        super().setup_class(self)
        self.model_path = os.path.join(self.test_samples, "tacotron2_statedict.pt")
        self.waveglow_path = os.path.join(self.test_samples, "waveglow_256channels_universal_v5.pt")
        self.graph_path = os.path.join(self.test_directory, "graph.png")
        self.audio_path = os.path.join(self.test_directory, "synthesized_audio.wav")

    def test_synthesize(self):
        model = load_model(self.model_path)
        assert model

        waveglow = load_waveglow(self.waveglow_path)
        assert waveglow

        text = "hello everybody my name is david attenborough"
        inflect_engine = inflect.engine()

        synthesize(model, waveglow, text, inflect_engine, graph=self.graph_path, audio=self.audio_path)

        assert text_similarity(text, transcribe(self.audio_path)) > 0.5
        assert os.path.isfile(self.graph_path)
        assert os.path.isfile(self.audio_path)
