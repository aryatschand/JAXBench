"""
Device Utilities

Utilities for device detection, synchronization, and management.
"""

from typing import List, Optional
import os


def get_available_devices() -> List[str]:
    """Get list of available JAX devices."""
    import jax
    return [str(d) for d in jax.devices()]


def is_tpu_available() -> bool:
    """Check if TPU is available."""
    import jax
    devices = jax.devices()
    return any("tpu" in str(d).lower() for d in devices)


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    import jax
    devices = jax.devices()
    return any("gpu" in str(d).lower() for d in devices)


def get_device_type() -> str:
    """Get the primary device type (tpu, gpu, or cpu)."""
    import jax
    device = jax.devices()[0]
    device_str = str(device).lower()
    if "tpu" in device_str:
        return "tpu"
    elif "gpu" in device_str:
        return "gpu"
    else:
        return "cpu"


def sync_device():
    """Synchronize device (wait for all operations to complete)."""
    import jax
    device_type = get_device_type()

    if device_type == "tpu":
        # For TPU, block_until_ready on a dummy computation
        x = jax.numpy.array([1.0])
        x.block_until_ready()
    elif device_type == "gpu":
        # For GPU, similar approach
        x = jax.numpy.array([1.0])
        x.block_until_ready()


def get_memory_info() -> dict:
    """Get device memory information (if available)."""
    try:
        import jax
        # This is platform-specific and may not work on all devices
        return {"available": True}
    except Exception:
        return {"available": False}


def set_device_count(count: int):
    """Set the number of devices to use (for multi-device setups)."""
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={count}"
