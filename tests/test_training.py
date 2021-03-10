import shutil
import inflect

from training.utils import check_space, CHECKPOINT_SIZE_MB
from training.clean_text import clean_text


def test_check_space_failure():
    _, _, free = shutil.disk_usage("/")
    free_mb = free // (2 ** 20)
    num_checkpoints = (free_mb // CHECKPOINT_SIZE_MB) + 1
    assert free_mb < CHECKPOINT_SIZE_MB * num_checkpoints
    exception = False
    try:
        check_space(num_checkpoints)
    except Exception as e:
        exception = True
        assert type(e) == AssertionError
    assert exception


def test_check_space_success():
    _, _, free = shutil.disk_usage("/")
    free_mb = free // (2 ** 20)
    num_checkpoints = (free_mb // CHECKPOINT_SIZE_MB) - 1
    assert free_mb > CHECKPOINT_SIZE_MB * num_checkpoints
    check_space(num_checkpoints)


def test_clean_text():
    text = clean_text("1st $500 Mr. 10.5 2,000 30 a\tb ~", inflect.engine())
    assert text == "first five hundred dollars mister ten point five two thousand thirty a b "
