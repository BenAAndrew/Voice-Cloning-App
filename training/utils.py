import shutil
import torch

CHECKPOINT_SIZE_MB = 333
BATCH_SIZE_PER_GB = 2.5
LEARNING_RATE_PER_BATCH = 3.125e-5


def get_available_memory():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    memory_in_use = torch.cuda.memory_allocated(0)
    available_memory = gpu_memory - memory_in_use
    available_memory_gb = available_memory // 1024 // 1024 // 1024
    return available_memory_gb


def get_batch_size(available_memory_gb):
    return int(available_memory_gb * BATCH_SIZE_PER_GB)


def get_learning_rate(batch_size):
    return batch_size * LEARNING_RATE_PER_BATCH


def check_space(num_checkpoints):
    _, _, free = shutil.disk_usage("/")
    free_mb = free // (2 ** 20)
    required_mb = CHECKPOINT_SIZE_MB * num_checkpoints
    assert (
        free_mb >= required_mb
    ), f"Insufficent storage space (requires {required_mb}mb). Reduce checkpoint frequency or free up space"
