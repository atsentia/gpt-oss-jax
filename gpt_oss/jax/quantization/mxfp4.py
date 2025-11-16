"""MXFP4 decompression with adaptive precision.

MXFP4 Format:
- 4-bit floating point with block-based exponent scaling
- 16-value FP4 lookup table
- 2 FP4 values packed per uint8 byte
- Scale factors biased by 127
- Used for MoE expert weights in GPT-OSS-20B
"""

import jax.numpy as jnp
import numpy as np
from typing import Literal

from .dtypes import get_target_dtype, TargetDtype


# FP4 lookup table (16 values: 8 positive, 8 negative)
FP4_VALUES = np.array([
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)


def decompress_mxfp4(
    blocks: np.ndarray,
    scales: np.ndarray,
    target_shape: tuple,
) -> jnp.ndarray:
    """Decompress MXFP4 quantized tensor to BF16 (backward compatible).

    This is the original function kept for backward compatibility.
    Use decompress_mxfp4_to_dtype() for adaptive precision.

    Args:
        blocks: MXFP4 blocks, shape [num_experts, out_dim, groups, 16]
        scales: Exponent scales, shape [num_experts, out_dim, groups]
        target_shape: Expected output shape [num_experts, out_dim, in_dim]

    Returns:
        Decompressed BF16 tensor of shape target_shape
    """
    return decompress_mxfp4_to_dtype(blocks, scales, target_shape, "bf16")


def decompress_mxfp4_to_dtype(
    blocks: np.ndarray,
    scales: np.ndarray,
    target_shape: tuple,
    target_dtype: TargetDtype = "bf16",
) -> jnp.ndarray:
    """Decompress MXFP4 quantized tensor to specified dtype.

    MXFP4 uses 4-bit floating point with block-based exponent scaling:
    1. Each uint8 byte contains 2 FP4 values (4 bits each)
    2. FP4 nibbles index into a 16-value lookup table (mantissas)
    3. Scales provide per-block exponent scaling (biased by 127)
    4. Final value = mantissa * 2^(scale - 127)

    Args:
        blocks: MXFP4 blocks, shape [num_experts, out_dim, groups, 16]
                Each uint8 contains 2 packed FP4 values
                groups = in_dim // 32, where 32 = 16 bytes * 2 FP4 values per byte
        scales: Exponent scales, shape [num_experts, out_dim, groups]
                Values are biased by 127
        target_shape: Expected output shape [num_experts, out_dim, in_dim]
        target_dtype: Target dtype ("bf16", "fp16", "fp8_e4m3fn", "fp8_e5m2")

    Returns:
        Decompressed tensor of shape target_shape with specified dtype

    Example:
        >>> blocks = np.array([[[[0x12, 0x34]]]], dtype=np.uint8)
        >>> scales = np.array([[[127]]], dtype=np.uint8)  # scale = 0 (unbiased)
        >>> output = decompress_mxfp4_to_dtype(blocks, scales, (1, 1, 4), "fp8_e4m3fn")
        >>> # Unpacks nibbles: [0x1, 0x2, 0x3, 0x4] → [+0.5, +1.0, +1.5, +2.0]
        >>> print(output.dtype)
        float8_e4m3fn
    """
    # Validate inputs
    assert blocks.dtype == np.uint8, \
        f"decompress_mxfp4: blocks must be uint8, got {blocks.dtype}"
    assert scales.dtype == np.uint8, \
        f"decompress_mxfp4: scales must be uint8, got {scales.dtype}"
    assert len(blocks.shape) == 4, \
        f"decompress_mxfp4: blocks must be 4D, got shape {blocks.shape}"
    assert len(scales.shape) == 3, \
        f"decompress_mxfp4: scales must be 3D, got shape {scales.shape}"
    assert len(target_shape) == 3, \
        f"decompress_mxfp4: target_shape must be 3D, got {target_shape}"

    num_experts, out_dim, groups, block_size = blocks.shape
    expected_in_dim = target_shape[2]

    assert block_size == 16, \
        f"decompress_mxfp4: expected block_size=16, got {block_size}"
    assert groups * block_size * 2 == expected_in_dim, \
        f"decompress_mxfp4: groups * 32 = {groups * 32} != target in_dim {expected_in_dim}"
    assert scales.shape == (num_experts, out_dim, groups), \
        f"decompress_mxfp4: scales shape {scales.shape} != expected {(num_experts, out_dim, groups)}"

    # Step 1: Unpack nibbles (each uint8 → 2 FP4 values)
    idx_lo = (blocks & 0x0F).astype(np.int32)  # Low nibble (bits 0-3)
    idx_hi = (blocks >> 4).astype(np.int32)    # High nibble (bits 4-7)

    # Step 2: Lookup mantissas from FP4 table
    mantissas_lo = FP4_VALUES[idx_lo]  # [num_experts, out_dim, groups, 16]
    mantissas_hi = FP4_VALUES[idx_hi]  # [num_experts, out_dim, groups, 16]

    # Step 3: Interleave mantissas: [lo[0], hi[0], lo[1], hi[1], ...]
    mantissas = np.empty((num_experts, out_dim, groups, block_size * 2), dtype=np.float32)
    mantissas[:, :, :, 0::2] = mantissas_lo
    mantissas[:, :, :, 1::2] = mantissas_hi
    # Shape: [num_experts, out_dim, groups, 32]

    # Step 4: Apply exponent scaling: value = mantissa * 2^(scale - 127)
    unbiased_scales = scales.astype(np.int32) - 127  # Remove bias
    scale_factors = np.power(2.0, unbiased_scales, dtype=np.float32)

    # Broadcast scales: [num_experts, out_dim, groups, 1]
    scale_factors = scale_factors[:, :, :, np.newaxis]

    # Apply scaling and reshape to target shape
    output_float32 = (mantissas * scale_factors).reshape(target_shape)

    # Step 5: Convert to target dtype
    output_dtype = get_target_dtype(target_dtype)

    return jnp.array(output_float32, dtype=output_dtype)


def get_mxfp4_block_info(blocks_shape: tuple) -> dict:
    """Get information about MXFP4 block structure.

    Args:
        blocks_shape: Shape of blocks tensor [num_experts, out_dim, groups, 16]

    Returns:
        Dictionary with block information

    Example:
        >>> info = get_mxfp4_block_info((128, 2880, 90, 16))
        >>> print(info)
        {'num_experts': 128, 'out_dim': 2880, 'in_dim': 2880, ...}
    """
    num_experts, out_dim, groups, block_size = blocks_shape

    return {
        "num_experts": num_experts,
        "out_dim": out_dim,
        "groups": groups,
        "block_size": block_size,
        "values_per_group": block_size * 2,  # 32 values (2 per byte)
        "in_dim": groups * block_size * 2,
        "compression_ratio": 4.0,  # 4-bit vs 16-bit FP
        "packed_size_mb": (num_experts * out_dim * groups * block_size) / 1e6,
    }
