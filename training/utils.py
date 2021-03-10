import shutil
import torch

CHECKPOINT_SIZE_MB = 333
BATCH_SIZE_PER_GB = 4
MINIMUM_MEMORY_GB = 4
LEARNING_RATE_PER_BATCH = 3.125e-5


def get_parameters():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    memory_in_use = torch.cuda.memory_allocated(0)
    available_memory = gpu_memory - memory_in_use
    available_memory_gb = available_memory // 1024 // 1024 // 1024
    assert (
        available_memory_gb >= MINIMUM_MEMORY_GB
    ), f"Required GPU with at least {MINIMUM_MEMORY_GB}GB memory. (only {available_memory_gb}GB available)"
    batch_size = (available_memory_gb - 1) * BATCH_SIZE_PER_GB
    learning_rate = batch_size * LEARNING_RATE_PER_BATCH
    return available_memory_gb, batch_size, learning_rate


def check_space(num_checkpoints):
    _, _, free = shutil.disk_usage("/")
    free_mb = free // (2 ** 20)
    required_mb = CHECKPOINT_SIZE_MB * num_checkpoints
    assert (
        free_mb >= required_mb
    ), f"Insufficent storage space (requires {required_mb}mb). Reduce checkpoint frequency or free up space"
