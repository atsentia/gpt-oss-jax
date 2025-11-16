"""Data type utilities for adaptive precision inference."""

import jax
import jax.numpy as jnp
from typing import Literal, Optional


# Supported target dtypes for decompression
TargetDtype = Literal["bf16", "fp16", "fp8_e4m3fn", "fp8_e5m2"]


def get_target_dtype(
    dtype_name: TargetDtype,
) -> jnp.dtype:
    """Get JAX dtype from string name.

    Args:
        dtype_name: Target dtype name ("bf16", "fp16", "fp8_e4m3fn", "fp8_e5m2")

    Returns:
        JAX dtype object

    Example:
        >>> dtype = get_target_dtype("fp8_e4m3fn")
        >>> print(dtype)
        float8_e4m3fn
    """
    dtype_map = {
        "bf16": jnp.bfloat16,
        "fp16": jnp.float16,
        "fp8_e4m3fn": jnp.float8_e4m3fn,  # 8-bit: 4-bit exponent, 3-bit mantissa
        "fp8_e5m2": jnp.float8_e5m2,      # 8-bit: 5-bit exponent, 2-bit mantissa
    }

    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_name}. "
            f"Supported: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_name]


def detect_tpu_type() -> tuple[Optional[str], int]:
    """Detect TPU type and number of devices.

    Returns:
        Tuple of (tpu_type, num_devices)
        - tpu_type: e.g., "TPU v2", "TPU v6e", or None if not TPU
        - num_devices: Number of TPU devices

    Example:
        >>> tpu_type, num_devices = detect_tpu_type()
        >>> print(f"{tpu_type} with {num_devices} devices")
        TPU v6e with 1 devices
    """
    backend = jax.default_backend()

    if backend != "tpu":
        return None, 0

    devices = jax.devices()
    tpu_type = devices[0].device_kind if devices else None

    return tpu_type, len(devices)


def recommend_dtype_for_tpu(
    tpu_type: Optional[str],
    num_devices: int,
) -> TargetDtype:
    """Recommend optimal dtype based on TPU type.

    Args:
        tpu_type: TPU device kind (e.g., "TPU v2", "TPU v6e")
        num_devices: Number of TPU devices

    Returns:
        Recommended dtype name

    Example:
        >>> dtype = recommend_dtype_for_tpu("TPU v6e", 1)
        >>> print(dtype)
        fp8_e4m3fn
    """
    if tpu_type is None:
        # CPU/GPU fallback
        return "bf16"

    # TPU v2-8: 64GB total HBM → use BF16
    if "v2" in tpu_type and num_devices == 8:
        return "bf16"

    # TPU v6e: ~32GB HBM → use FP8 for memory efficiency
    elif "v6" in tpu_type:
        return "fp8_e4m3fn"

    # TPU v5e: Limited memory → use FP8
    elif "v5" in tpu_type:
        return "fp8_e4m3fn"

    # Default to BF16 for unknown TPU types
    else:
        return "bf16"


def estimate_memory_usage(params, unit: str = "GB") -> float:
    """Estimate memory usage of parameter tree.

    Args:
        params: Flax parameter tree
        unit: Unit for output ("GB", "MB", "KB", "B")

    Returns:
        Memory usage in specified unit

    Example:
        >>> memory_gb = estimate_memory_usage(params, "GB")
        >>> print(f"Model uses {memory_gb:.2f} GB")
        Model uses 21.45 GB
    """
    total_bytes = sum(
        p.nbytes for p in jax.tree_util.tree_leaves(params)
    )

    units = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}

    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}. Supported: {list(units.keys())}")

    return total_bytes / units[unit]


def get_dtype_info(dtype_name: TargetDtype) -> dict:
    """Get information about a dtype.

    Args:
        dtype_name: Target dtype name

    Returns:
        Dictionary with dtype information

    Example:
        >>> info = get_dtype_info("fp8_e4m3fn")
        >>> print(info)
        {'bits': 8, 'name': 'FP8 E4M3', 'description': '...'}
    """
    dtype_info = {
        "bf16": {
            "bits": 16,
            "name": "BFloat16",
            "description": "Brain Float 16: 8-bit exponent, 7-bit mantissa",
            "range": "±3.4e38",
            "precision": "~3 decimal digits",
        },
        "fp16": {
            "bits": 16,
            "name": "Float16",
            "description": "IEEE 754 Half Precision: 5-bit exponent, 10-bit mantissa",
            "range": "±6.5e4",
            "precision": "~3 decimal digits",
        },
        "fp8_e4m3fn": {
            "bits": 8,
            "name": "FP8 E4M3",
            "description": "8-bit Float: 4-bit exponent, 3-bit mantissa, no infinities",
            "range": "±448",
            "precision": "~2 decimal digits",
        },
        "fp8_e5m2": {
            "bits": 8,
            "name": "FP8 E5M2",
            "description": "8-bit Float: 5-bit exponent, 2-bit mantissa",
            "range": "±5.7e4",
            "precision": "~1 decimal digit",
        },
    }

    return dtype_info.get(dtype_name, {})
