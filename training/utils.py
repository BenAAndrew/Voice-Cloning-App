import shutil
import torch

CHECKPOINT_SIZE_MB = 333
BATCH_SIZE_PER_GB = 2.5
LEARNING_RATE_PER_BATCH = 3.125e-5


def get_available_memory():
    """
    Get available GPU memory in GB.

    Returns
    -------
    int
        Available GPU memory in GB
    """
    available_memory_gb = 0

    for i in range(torch.cuda.device_count()):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory
        memory_in_use = torch.cuda.memory_allocated(i)
        available_memory = gpu_memory - memory_in_use
        available_memory_gb += available_memory // 1024 // 1024 // 1024

    return available_memory_gb


def get_batch_size(available_memory_gb):
    """
    Calulate batch size.

    Parameters
    ----------
    available_memory_gb : int
        Available GPU memory in GB

    Returns
    -------
    int
        Batch size
    """
    return int(available_memory_gb * BATCH_SIZE_PER_GB)


def get_learning_rate(batch_size):
    """
    Calulate learning rate.

    Parameters
    ----------
    batch_size : int
        Batch size

    Returns
    -------
    float
        Learning rate
    """
    return batch_size * LEARNING_RATE_PER_BATCH


def check_space(num_checkpoints):
    """
    Check if system has enough available storage to save all checkpoints.

    Parameters
    ----------
    num_checkpoints : int
        Number of checkpoints that will be generated in training

    Raises
    -------
    AssertionError
        If system does not have sufficent storage space
    """
    _, _, free = shutil.disk_usage("/")
    free_mb = free // (2 ** 20)
    required_mb = CHECKPOINT_SIZE_MB * num_checkpoints
    assert (
        free_mb >= required_mb
    ), f"Insufficent storage space (requires {required_mb}mb). Reduce checkpoint frequency or free up space"
