#!/usr/bin/env python3
"""
Test FP8 conversion on Mac M3 Ultra with RAM monitoring.

This script simulates TPU v6e FP8 conversion locally to verify:
1. Path configuration works correctly
2. SafeTensors loading succeeds
3. BF16 → FP8 conversion works
4. RAM usage is tracked at each step
5. Orbax save succeeds

Usage:
    python scripts/test_fp8_conversion_mac.py

    # Or with custom path:
    python scripts/test_fp8_conversion_mac.py --path /path/to/gpt-oss-20b/original
"""

import argparse
import os
import sys
import time
import gc
import psutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from gpt_oss.jax.config import ModelConfig
from gpt_oss.jax.loader_safetensors import WeightLoader
import orbax.checkpoint as ocp


def get_ram_gb():
    """Get current RAM usage in GB."""
    return psutil.Process().memory_info().rss / 1024**3


def main():
    parser = argparse.ArgumentParser(description="Test FP8 conversion on Mac")
    parser.add_argument(
        "--path",
        type=str,
        default="gpt-oss-20b/original",
        help="Path to gpt-oss-20b SafeTensors directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Orbax checkpoint directory (default: ./gpt-oss-20b-orbax-fp8)"
    )
    args = parser.parse_args()

    # Convert paths to absolute
    safetensors_path = Path(args.path).resolve()
    if args.output:
        orbax_path = str(Path(args.output).resolve())
    else:
        orbax_path = str((Path.cwd() / "gpt-oss-20b-orbax-fp8").resolve())

    print("=" * 70)
    print("FP8 CONVERSION TEST - MAC M3 ULTRA")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Setup Environment
    # =========================================================================
    print("\n[1/5] Setting up environment...")

    # Force JAX to use CPU
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['COLAB_TPU_ADDR'] = 'fake-tpu-v6e'  # Trigger FP8 logic

    print(f"  JAX devices: {jax.devices()}")
    print(f"  JAX backend: {jax.default_backend()}")

    STRATEGY = "fp8"
    DTYPE = jnp.float8_e4m3fn
    print(f"  Strategy: {STRATEGY.upper()}")
    print(f"  Target dtype: {DTYPE}")

    # =========================================================================
    # STEP 2: Validate Paths
    # =========================================================================
    print("\n[2/5] Validating paths...")

    if not safetensors_path.exists():
        print(f"  ✗ ERROR: Path does not exist: {safetensors_path}")
        print(f"\nPlease provide correct path:")
        print(f"  python {sys.argv[0]} --path /path/to/gpt-oss-20b/original")
        sys.exit(1)

    st_files = list(safetensors_path.glob('*.safetensors'))
    if len(st_files) == 0:
        # Try common subdirectories
        for subdir in ['original', 'models']:
            candidate = safetensors_path / subdir
            st_files = list(candidate.glob('*.safetensors'))
            if st_files:
                safetensors_path = candidate
                print(f"  Auto-discovered: {safetensors_path}")
                break

        if len(st_files) == 0:
            print(f"  ✗ ERROR: No .safetensors files found in: {safetensors_path}")
            sys.exit(1)

    print(f"  ✓ SafeTensors path: {safetensors_path}")
    print(f"  ✓ Found {len(st_files)} .safetensors file(s)")
    print(f"  ✓ Orbax output: {orbax_path}")

    # =========================================================================
    # STEP 3: Load Model Configuration
    # =========================================================================
    print("\n[3/5] Loading model configuration...")

    config = ModelConfig()
    print(f"  Model: GPT-OSS-20B")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Parameters: ~21B")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  MoE experts: {config.num_experts}")

    # =========================================================================
    # STEP 4: Convert SafeTensors → Orbax (FP8)
    # =========================================================================
    print("\n[4/5] Converting to Orbax FP8 format...")
    print("  This will take 2-5 minutes...")

    # Clean up existing checkpoint if it exists
    if Path(orbax_path).exists():
        print(f"  Removing existing checkpoint: {orbax_path}")
        import shutil
        shutil.rmtree(orbax_path)

    ram_start = get_ram_gb()
    print(f"  Starting RAM: {ram_start:.2f} GB")
    t0 = time.time()

    # Load and convert on CPU
    with jax.default_device(jax.devices('cpu')[0]):
        # Load weights directly as FP8 (MXFP4 → FP8, skip BF16 intermediate)
        print(f"\n  Loading SafeTensors directly as FP8...")
        print(f"  (MXFP4 weights decompress directly to FP8, BF16 weights convert to FP8)")
        loader = WeightLoader(str(safetensors_path))
        params = loader.load_params(config, target_dtype=DTYPE, show_progress=True)

        ram_after_load = get_ram_gb()
        delta_load = ram_after_load - ram_start
        print(f"\n  After loading FP8: {ram_after_load:.2f} GB (+{delta_load:.2f} GB)")

    # Save to Orbax
    print(f"\n  Saving to Orbax checkpoint...")
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(orbax_path, params)

    ram_peak = get_ram_gb()
    delta_peak = ram_peak - ram_start

    # Cleanup
    del params
    gc.collect()
    ram_after_cleanup = get_ram_gb()
    freed = ram_peak - ram_after_cleanup

    elapsed = time.time() - t0

    print(f"\n  ✓ Conversion complete in {elapsed:.1f}s")
    print(f"  ✓ Orbax checkpoint: {orbax_path}")

    # =========================================================================
    # STEP 5: Summary
    # =========================================================================
    print("\n[5/5] Summary")
    print("=" * 70)
    print(f"  Conversion time: {elapsed:.1f}s")
    print(f"  Peak RAM usage: {ram_peak:.2f} GB (+{delta_peak:.2f} GB)")
    print(f"  After cleanup: {ram_after_cleanup:.2f} GB ({freed:.2f} GB freed)")
    print(f"\n  RAM breakdown:")
    print(f"    - Starting:        {ram_start:.2f} GB")
    print(f"    - After FP8 load:  {ram_after_load:.2f} GB (+{delta_load:.2f} GB)")
    print(f"    - Peak (save):     {ram_peak:.2f} GB (+{delta_peak:.2f} GB)")
    print(f"\n  Expected ranges (direct MXFP4→FP8 conversion):")
    print(f"    - FP8 load:  +20-22 GB (21B params × 1 byte, no BF16 intermediate)")
    print(f"    - Peak:      +25-30 GB (temporary during Orbax save)")

    print("\n" + "=" * 70)
    print("✓ FP8 CONVERSION TEST PASSED")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Verify Orbax checkpoint: ls -lh {orbax_path}")
    print(f"  2. Test loading: python -c 'import orbax.checkpoint as ocp; ...")
    print(f"  3. Upload to Colab or run inference locally")


if __name__ == "__main__":
    main()
