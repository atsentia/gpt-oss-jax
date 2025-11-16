#!/usr/bin/env python3
"""Test mixed-precision Orbax checkpoint loading and Harmony protocol inference.

This test validates that:
1. Mixed-precision Orbax checkpoints load correctly (BF16 + FP8)
2. Automatic FP8→float32 upcasting works in inference
3. Harmony protocol (dual-channel reasoning) works with mixed-precision weights
4. Generated output is valid

Mixed Precision Strategy:
- BF16 params (15.1%): Embedding, norms, attention, gates, biases → stay BF16
- FP8 params (84.9%): MoE expert weights → decompress MXFP4 to FP8
- Benefits: Better quality than full FP8, fits in TPU v6e 32 GB HBM

Usage:
    python tests/test_fp8_harmony_inference.py
"""

import sys
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt_oss.jax.model import Transformer
from gpt_oss.jax.config import ModelConfig
from gpt_oss.jax.inference import generate
from gpt_oss.jax.tokenizer import get_tokenizer


def test_fp8_checkpoint_loading():
    """Test that mixed-precision Orbax checkpoint loads correctly."""
    print("\n" + "="*80)
    print("TEST 1: Mixed-Precision Orbax Checkpoint Loading")
    print("="*80)

    checkpoint_path = Path("/Users/amund/jax-for-gpt-oss/gpt-oss-20b-orbax-mixed")

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ FAILED: Checkpoint not found at {checkpoint_path}")
        print(f"   Please run scripts/test_fp8_conversion_mac.py first to create the checkpoint")
        return False

    print(f"✓ Checkpoint found: {checkpoint_path}")

    # Load checkpoint
    try:
        checkpointer = ocp.StandardCheckpointer()
        params = checkpointer.restore(checkpoint_path)
        print(f"✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ FAILED: Error loading checkpoint: {e}")
        return False

    # Verify FP8 dtype
    def count_dtypes(tree):
        """Count parameters by dtype."""
        dtypes = {}

        def count(x):
            if isinstance(x, jax.Array):
                dtype_name = str(x.dtype)
                dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1

        jax.tree_util.tree_map(count, tree)
        return dtypes

    dtype_counts = count_dtypes(params)
    print(f"✓ Parameter dtypes: {dtype_counts}")

    # Check that we have both BF16 and FP8 parameters (mixed precision)
    if 'bfloat16' not in dtype_counts:
        print(f"❌ FAILED: No BF16 parameters found (expected bfloat16)")
        return False

    if 'float8_e4m3fn' not in dtype_counts:
        print(f"❌ FAILED: No FP8 parameters found (expected float8_e4m3fn)")
        return False

    print(f"✓ BF16 parameters detected: {dtype_counts.get('bfloat16', 0)} arrays (small params)")
    print(f"✓ FP8 parameters detected: {dtype_counts['float8_e4m3fn']} arrays (expert weights)")
    print("\n✅ TEST 1 PASSED: Mixed-precision checkpoint loading works\n")
    return True


def test_fp8_harmony_inference():
    """Test Harmony protocol inference with mixed-precision weights."""
    print("\n" + "="*80)
    print("TEST 2: Harmony Protocol Inference with Mixed-Precision Weights")
    print("="*80)

    # Configuration
    checkpoint_path = Path("/Users/amund/jax-for-gpt-oss/gpt-oss-20b-orbax-mixed")

    # Load model config (use defaults, which match the checkpoint)
    config = ModelConfig()

    # Create model
    print(f"Creating Transformer model...")
    model = Transformer(config=config)
    print(f"✓ Model created")

    # Load mixed-precision checkpoint
    print(f"Loading mixed-precision checkpoint from {checkpoint_path}...")
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(checkpoint_path)
    print(f"✓ Checkpoint loaded")

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"✓ Tokenizer loaded")

    # Harmony protocol test prompt
    user_query = "What is the capital of France?"

    # Format Harmony protocol prompt (dual-channel reasoning)
    harmony_prompt = f"""<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
<analysis>"""

    print(f"\n" + "-"*80)
    print(f"User query: {user_query}")
    print(f"-"*80)

    # Tokenize
    prompt_tokens = tokenizer.encode(harmony_prompt)
    print(f"✓ Tokenized: {len(prompt_tokens)} tokens")

    # Generate (automatic FP8→float32 upcasting happens here)
    print(f"\nGenerating analysis channel (30 tokens)...")
    try:
        rng_key = jax.random.PRNGKey(42)
        output_tokens, stats = generate(
            model=model,
            params=params,
            prompt_tokens=prompt_tokens,
            max_new_tokens=30,
            temperature=0.7,
            rng_key=rng_key,
            show_progress=False,
            return_stats=True,
            use_kv_cache=True,
            config=config
        )
        print(f"✓ Generation completed")
    except Exception as e:
        print(f"❌ FAILED: Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Decode output
    output_text = tokenizer.decode(output_tokens)
    analysis_text = output_text.split("<analysis>")[-1].split("</analysis>")[0] if "</analysis>" in output_text else output_text.split("<analysis>")[-1]

    print(f"\n" + "-"*80)
    print(f"Analysis channel output:")
    print(f"-"*80)
    print(analysis_text)
    print(f"-"*80)

    # Verify stats
    print(f"\nPerformance stats:")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Time to first token: {stats['first_token_time']:.2f}s")
    print(f"  Tokens generated: {stats['num_tokens']}")
    print(f"  Tokens/second: {stats['tokens_per_second']:.2f}")
    print(f"  Tokens/second (after first): {stats['tokens_per_second_after_first']:.2f}")

    # Validate output
    if len(output_tokens) <= len(prompt_tokens):
        print(f"❌ FAILED: No new tokens generated")
        return False

    if stats['num_tokens'] != 30:
        print(f"❌ FAILED: Expected 30 tokens, got {stats['num_tokens']}")
        return False

    print(f"\n✅ TEST 2 PASSED: Harmony inference with FP8 works\n")
    return True


def test_fp8_automatic_upcasting():
    """Test that FP8→float32 upcasting happens automatically (preserves BF16)."""
    print("\n" + "="*80)
    print("TEST 3: Automatic FP8→float32 Upcasting (Mixed Precision)")
    print("="*80)

    checkpoint_path = Path("/Users/amund/jax-for-gpt-oss/gpt-oss-20b-orbax-mixed")

    # Load mixed-precision checkpoint
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(checkpoint_path)

    # Check BF16 parameter (attention weights)
    bf16_param = params['block_0']['attn']['qkv']['kernel']
    print(f"BF16 param dtype: {bf16_param.dtype}")

    if bf16_param.dtype != jnp.bfloat16:
        print(f"❌ FAILED: Expected bfloat16 for attention, got {bf16_param.dtype}")
        return False

    # Check FP8 parameter (expert weights)
    fp8_param = params['block_0']['mlp']['mlp1_weight']
    print(f"FP8 param dtype: {fp8_param.dtype}")

    if fp8_param.dtype != jnp.float8_e4m3fn:
        print(f"❌ FAILED: Expected float8_e4m3fn for experts, got {fp8_param.dtype}")
        return False

    print(f"✓ Confirmed BF16 parameters (attention)")
    print(f"✓ Confirmed FP8 parameters (experts)")

    # Import upcast function
    from gpt_oss.jax.model import upcast_fp8_params

    # Apply upcasting
    params_upcast = upcast_fp8_params(params)

    # Check that BF16 parameters stay BF16
    bf16_param_upcast = params_upcast['block_0']['attn']['qkv']['kernel']
    print(f"BF16 after upcasting: {bf16_param_upcast.dtype}")

    if bf16_param_upcast.dtype != jnp.bfloat16:
        print(f"❌ FAILED: BF16 should stay BF16, got {bf16_param_upcast.dtype}")
        return False

    # Check that FP8 parameters upcast to float32
    fp8_param_upcast = params_upcast['block_0']['mlp']['mlp1_weight']
    print(f"FP8 after upcasting: {fp8_param_upcast.dtype}")

    if fp8_param_upcast.dtype != jnp.float32:
        print(f"❌ FAILED: Expected float32 after upcasting FP8, got {fp8_param_upcast.dtype}")
        return False

    print(f"✓ Confirmed BF16 preserved (not upcast)")
    print(f"✓ Confirmed FP8 upcast to float32")

    # Verify FP8 values are preserved (within FP8 precision)
    # Convert original FP8 to float32 for comparison
    original_as_f32 = fp8_param.astype(jnp.float32)
    max_diff = jnp.max(jnp.abs(original_as_f32 - fp8_param_upcast))

    print(f"✓ Max difference after FP8 upcasting: {max_diff:.2e} (should be 0.0)")

    if max_diff > 1e-10:
        print(f"❌ FAILED: Upcasting changed FP8 values (max diff: {max_diff})")
        return False

    print(f"\n✅ TEST 3 PASSED: Automatic upcasting preserves values\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("FP8 Harmony Inference Test Suite")
    print("="*80)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print("="*80)

    results = []

    # Test 1: Checkpoint loading
    results.append(("FP8 Checkpoint Loading", test_fp8_checkpoint_loading()))

    # Test 2: Harmony inference
    results.append(("Harmony Protocol Inference", test_fp8_harmony_inference()))

    # Test 3: Automatic upcasting
    results.append(("Automatic FP8→float32 Upcasting", test_fp8_automatic_upcasting()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("="*80)

    # Exit code
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed!\n")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
