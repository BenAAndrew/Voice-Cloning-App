import os
import json
from pydub import AudioSegment

from dataset.clip_generator import (
    clip_generator,
    load_audio,
    load_forced_alignment_data,
    create_audio_snippet,
)
from dataset.transcribe import transcribe
from dataset.forced_alignment.align import align, MIN_SWS
from tests.utils import TestClass, text_similarity


class TestClipGenerator(TestClass):
    def setup_class(self):
        super().setup_class(self)
        self.forced_alignment_path = os.path.join(self.test_directory, "align.json")
        self.output_directory = os.path.join(self.test_directory, "output")
        self.label_path = os.path.join(self.test_directory, "label.txt")
        self.audio_path = os.path.join(self.test_samples, "audio.wav")
        with open(self.forced_alignment_path, "w") as forced_alignment_file:
            forced_alignment_file.write(
                '[{"start": 0, "end": 2000, "aligned": "Hello"}, {"start": 2500, "end": 5000, "aligned": "world"}]'
            )

    def test_load_audio(self):
        audio = load_audio(self.audio_path, 22050)
        assert len(audio) == 7584

    def test_load_forced_alignment_data(self):
        sentences = load_forced_alignment_data(self.forced_alignment_path)
        assert len(sentences) == 2
        assert sentences[0] == {"start": 0, "end": 2000, "aligned": "Hello"}
        assert sentences[1] == {"start": 2500, "end": 5000, "aligned": "world"}

    def test_create_audio_snippet(self):
        audio = load_audio(self.audio_path, 22050)
        silence = AudioSegment.silent(duration=200)
        name = create_audio_snippet(audio, 0, 5000, silence, self.test_directory)
        path = os.path.join(self.test_directory, name)

        assert name == "0_5000.wav"
        assert os.path.isfile(path)
        clip = load_audio(path, 22050)
        assert len(clip) == 5200

    def test_clip_generator(self):
        clip_generator(
            audio_path=self.audio_path,
            forced_alignment_path=self.forced_alignment_path,
            output_path=self.output_directory,
            label_path=self.label_path,
        )

        assert os.listdir(self.output_directory) == ["0_2000.wav", "2500_5000.wav"]
        with open(self.label_path) as f:
            assert f.read() == "0_2000.wav|Hello\n2500_5000.wav|world\n"


class TestForcedAlignment(TestClass):
    def setup_class(self):
        super().setup_class(self)
        self.audio_path = os.path.join(self.test_samples, "audio.wav")
        self.text_path = os.path.join(self.test_samples, "text.txt")
        self.forced_alignment_path = os.path.join(self.test_directory, "align.json")

    def test_transcription(self):
        text = transcribe(self.audio_path)
        expected = "the examination and testimony of the experts enabled the commission to conclude that five shots may have been fired"
        assert text_similarity(expected, text) > 0.5

    def test_force_alignment(self):
        output_path = os.path.join(self.test_directory, "align.json")
        align(self.audio_path, self.text_path, output_path)
        assert os.path.isfile(output_path)

        with open(output_path, "r") as forced_alignment_file:
            data = json.load(forced_alignment_file)
            sentences = [
                "the examination and testimony of the experts",
                "enabled the commission to conclude",
                "that five shots may have been fired",
            ]
            assert len(data) == 3
            for i in range(3):
                assert data[i]["aligned"] == sentences[i]
                assert data[i]["sws"] > MIN_SWS
                assert set(["start", "end", "transcript"]).issubset(data[i].keys())
