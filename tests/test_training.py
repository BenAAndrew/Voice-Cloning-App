import shutil
import inflect
from unittest import mock

from training.utils import check_space, CHECKPOINT_SIZE_MB
from training.clean_text import clean_text


@mock.patch('shutil.disk_usage')
def test_check_space_failure(disk_usage):
    disk_usage.return_value = None, None, (CHECKPOINT_SIZE_MB) * (2 ** 20)
    exception = False
    try:
        check_space(2)
    except Exception as e:
        exception = True
        assert type(e) == AssertionError
    assert exception, "Insufficent space should throw an exception"


@mock.patch('shutil.disk_usage')
def test_check_space_success(disk_usage):
    disk_usage.return_value = None, None, (CHECKPOINT_SIZE_MB + 1) * (2 ** 20)
    assert check_space(1) is None, "Sufficent space should not throw an exception"


def test_clean_text():
    text = clean_text("1st $500 Mr. 10.5 2,000 30 a\tb ~", inflect.engine())
    assert text == "first five hundred dollars mister ten point five two thousand thirty a b "
