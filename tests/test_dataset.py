import os
import json
from pydub import AudioSegment

from dataset.transcribe import transcribe
from dataset.clip_generator import clip_generator
from tests.utils import TestClass


class TestClipGenerator(TestClass):
    def setup_class(self):
        super().setup_class(self)
        self.audio_path = os.path.join(self.test_samples, "audio.wav")
        self.script_path = os.path.join(self.test_samples, "text.txt")
        self.forced_alignment_path = os.path.join(self.test_directory, "align.json")
        self.output_directory = os.path.join(self.test_directory, "wavs")
        self.label_path = os.path.join(self.test_directory, "metadata.csv")

    def test_clip_generator(self):
        min_confidence = 0.85
        clip_generator(
            audio_path=self.audio_path,
            script_path=self.script_path,
            forced_alignment_path=self.forced_alignment_path,
            output_path=self.output_directory,
            label_path=self.label_path,
            min_confidence=min_confidence
        )

        assert os.listdir(self.output_directory) == ["0_2730.wav", "2820_5100.wav", "5130_7560.wav"], "Unexpected audio clips"
        
        with open(self.label_path) as f:
            lines = f.readlines()
            assert lines == [
                "0_2730.wav|the examination and testimony of the experts\n",
                "2820_5100.wav|enabled the commission to conclude\n",
                "5130_7560.wav|that five shots may have been fired\n"
            ], "Unexpected metadata contents"

        with open(self.forced_alignment_path, "r") as forced_alignment_file:
            data = json.load(forced_alignment_file)
            for segment in data:
                assert {"start", "end", "name", "sws", "aligned"}.issubset(segment.keys()), "Alignment JSON missing required keys"
                assert segment["sws"] >= min_confidence, "SWS score less than min confidence"
