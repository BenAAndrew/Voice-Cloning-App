import os
import shutil


class TestClass:
    test_directory = "test"
    test_samples = "test_samples"

    def setup_class(self):
        if os.path.isdir(self.test_directory):
            shutil.rmtree(self.test_directory)

        assert os.path.isdir(self.test_samples), f"Must add test samples to {test_samples}"
        os.makedirs(self.test_directory, exist_ok=False)

    def teardown_class(self):
        shutil.rmtree(self.test_directory)


def text_similarity(a, b):
    return 1 - (len(set(a.split(" ")) - set(b.split(" "))) / len(a.split(" ")))
