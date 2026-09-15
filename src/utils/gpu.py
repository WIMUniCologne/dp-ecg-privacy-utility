"""
GPU memory management for parallel training runs.

By default TensorFlow grabs all available GPU memory. To run multiple
training processes on the same GPU concurrently, each process must
limit its memory consumption.

Usage:
    from src.utils.gpu import limit_gpu_memory
    limit_gpu_memory(memory_gb=8.0)   # at the top of your script

Call this BEFORE any other TensorFlow operation (including imports that
might initialize TF).
"""
from __future__ import annotations

import os


def limit_gpu_memory(memory_gb: float = 8.0) -> None:
    """
    Limit TensorFlow's GPU memory usage to `memory_gb` GiB.

    Useful when running multiple training processes on the same GPU
    in parallel.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    memory_mb = int(memory_gb * 1024)
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_mb)],
            )
    except RuntimeError as e:
        # Already initialized — too late
        print(f"WARNING: GPU memory limit could not be set: {e}")


def enable_gpu_memory_growth() -> None:
    """
    Alternative to limit_gpu_memory: grow memory on-demand instead of
    pre-allocating. Less predictable for parallel runs but simpler.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"WARNING: GPU memory growth could not be enabled: {e}")
