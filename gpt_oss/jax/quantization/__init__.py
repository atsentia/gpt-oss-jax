"""Quantization utilities for JAX backend.

This module provides utilities for working with quantized weights:
- MXFP4 decompression (4-bit → FP8/BF16)
- Mixed-precision conversion helpers
- Dtype utilities for different hardware targets
"""

from .mxfp4 import decompress_mxfp4, decompress_mxfp4_to_dtype
from .dtypes import get_target_dtype, estimate_memory_usage

__all__ = [
    "decompress_mxfp4",
    "decompress_mxfp4_to_dtype",
    "get_target_dtype",
    "estimate_memory_usage",
]
