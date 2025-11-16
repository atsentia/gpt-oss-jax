#!/usr/bin/env python3
"""
Analyze dtypes and sizes in a SafeTensors checkpoint.

Usage:
    python scripts/analyze_safetensors_dtypes.py gpt-oss-20b/original
"""

import argparse
from pathlib import Path
from safetensors import safe_open
import numpy as np


def bytes_to_gb(bytes_val):
    """Convert bytes to GB."""
    return bytes_val / (1024 ** 3)


def analyze_checkpoint(checkpoint_path: str):
    """Analyze SafeTensors checkpoint dtypes and sizes."""
    path = Path(checkpoint_path)

    # Find all .safetensors files
    st_files = list(path.glob('*.safetensors'))
    if not st_files:
        print(f"Error: No .safetensors files found in {checkpoint_path}")
        return

    print("=" * 70)
    print(f"SAFETENSORS CHECKPOINT ANALYSIS")
    print("=" * 70)
    print(f"Path: {checkpoint_path}")
    print(f"Files: {len(st_files)}")
    print()

    # Collect stats by dtype
    dtype_stats = {}
    total_params = 0
    total_bytes = 0

    for st_file in st_files:
        with safe_open(str(st_file), framework='np', device='cpu') as f:
            for key in f.keys():
                # Get metadata without loading tensor (BF16 not supported by numpy)
                metadata = f.get_slice(key)
                shape = metadata.get_shape()
                dtype = str(metadata.get_dtype())

                num_params = np.prod(shape)
                # Calculate bytes based on dtype
                bytes_per_elem = {
                    'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8,
                    'I8': 1, 'I16': 2, 'I32': 4, 'I64': 8,
                    'U8': 1, 'U16': 2, 'U32': 4, 'U64': 8
                }.get(dtype, 4)  # Default to 4 bytes if unknown
                num_bytes = num_params * bytes_per_elem

                if dtype not in dtype_stats:
                    dtype_stats[dtype] = {
                        'count': 0,
                        'params': 0,
                        'bytes': 0,
                        'tensors': []
                    }

                dtype_stats[dtype]['count'] += 1
                dtype_stats[dtype]['params'] += num_params
                dtype_stats[dtype]['bytes'] += num_bytes
                dtype_stats[dtype]['tensors'].append(key)

                total_params += num_params
                total_bytes += num_bytes

    # Print summary
    print(f"{'Dtype':<12} {'Tensors':<10} {'Parameters':<15} {'Size (GB)':<12} {'% Params':<10}")
    print("-" * 70)

    for dtype in sorted(dtype_stats.keys()):
        stats = dtype_stats[dtype]
        count = stats['count']
        params = stats['params']
        size_gb = bytes_to_gb(stats['bytes'])
        pct = (params / total_params) * 100

        print(f"{dtype:<12} {count:<10} {params:>14,} {size_gb:>11.2f} {pct:>9.1f}%")

    print("-" * 70)
    print(f"{'TOTAL':<12} {sum(s['count'] for s in dtype_stats.values()):<10} "
          f"{total_params:>14,} {bytes_to_gb(total_bytes):>11.2f} {'100.0':>9}%")

    print()
    print("=" * 70)
    print("DTYPE EXPLANATION")
    print("=" * 70)
    print("  uint8:    8-bit unsigned integer (used for MXFP4 blocks and scales)")
    print("  bfloat16: 16-bit brain floating point (most parameters)")
    print("  float32:  32-bit floating point (some special parameters)")
    print()
    print("MXFP4 Format:")
    print("  - Stored as uint8 (blocks + scales)")
    print("  - Decompresses to BF16 or FP8 during loading")
    print("  - 2 bytes per parameter (4 bits packed + scale overhead)")
    print()

    # Show example MXFP4 parameters
    mxfp4_params = []
    for dtype, stats in dtype_stats.items():
        if dtype == 'uint8':
            for tensor_name in stats['tensors'][:5]:
                if '.blocks' in tensor_name or '.scales' in tensor_name:
                    mxfp4_params.append(tensor_name)

    if mxfp4_params:
        print("Example MXFP4 Parameters:")
        for param in mxfp4_params[:3]:
            print(f"  - {param}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SafeTensors checkpoint dtypes")
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint_path)
