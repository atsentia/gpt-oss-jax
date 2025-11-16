"""Example: Using JAX optimizations for efficient inference.

This example demonstrates how to enable and benchmark different JAX optimizations
for the gpt-oss-20b model.

Run this example:
    python examples/optimization_demo.py
"""

import jax
import jax.numpy as jnp
from gpt_oss.jax.model import Transformer
from gpt_oss.jax.config import ModelConfig
from gpt_oss.jax.inference import generate
from gpt_oss.jax.sharding import (
    create_device_mesh,
    get_data_parallel_sharding,
    print_sharding_info
)
import time


def demo_baseline_config():
    """Baseline configuration (no optimizations)."""
    print("\n" + "="*80)
    print("BASELINE CONFIGURATION")
    print("="*80)
    
    config = ModelConfig(
        # Smaller model for demo
        num_hidden_layers=2,
        num_experts=8,
        experts_per_token=2,
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        head_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        
        # No optimizations
        optimize_gqa=False,
        optimize_moe=False,
        use_scan_layers=False,
    )
    
    print("\nConfiguration:")
    print(f"  optimize_gqa: {config.optimize_gqa}")
    print(f"  optimize_moe: {config.optimize_moe}")
    print(f"  use_scan_layers: {config.use_scan_layers}")
    
    return config


def demo_optimized_config():
    """Optimized configuration (all optimizations enabled)."""
    print("\n" + "="*80)
    print("OPTIMIZED CONFIGURATION")
    print("="*80)
    
    config = ModelConfig(
        # Same model size
        num_hidden_layers=2,
        num_experts=8,
        experts_per_token=2,
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        head_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        
        # All optimizations enabled
        optimize_gqa=True,
        optimize_moe=True,
        use_scan_layers=False,  # Not yet implemented
    )
    
    print("\nConfiguration:")
    print(f"  optimize_gqa: {config.optimize_gqa}")
    print(f"  optimize_moe: {config.optimize_moe}")
    print(f"  use_scan_layers: {config.use_scan_layers}")
    
    return config


def demo_generation(config, name):
    """Demo generation with the given configuration."""
    print(f"\n{'-'*80}")
    print(f"Testing: {name}")
    print(f"{'-'*80}")
    
    # Create model
    model = Transformer(
        config=config,
        optimize_gqa=config.optimize_gqa,
        optimize_moe=config.optimize_moe,
    )
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    prompt = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    
    print("\n1. Initializing model...")
    init_start = time.time()
    params = model.init(key, prompt)['params']
    init_time = time.time() - init_start
    print(f"   Initialization time: {init_time:.2f}s")
    
    # Forward pass
    print("\n2. Running forward pass...")
    forward_start = time.time()
    logits = model.apply({'params': params}, prompt)
    logits[-1].block_until_ready()  # Wait for computation
    forward_time = time.time() - forward_start
    print(f"   Forward pass time: {forward_time*1000:.2f}ms")
    print(f"   Output shape: {logits.shape}")
    
    # Generation (with KV cache)
    print("\n3. Generating tokens (with KV cache)...")
    gen_start = time.time()
    tokens, stats = generate(
        model=model,
        params=params,
        prompt_tokens=[1, 2, 3, 4, 5],
        max_new_tokens=10,
        temperature=0.0,
        show_progress=False,
        return_stats=True,
        use_kv_cache=True,
        config=config
    )
    gen_time = time.time() - gen_start
    
    print(f"   Total generation time: {gen_time:.2f}s")
    print(f"   First token time: {stats['first_token_time']:.2f}s")
    print(f"   Tokens/second: {stats['tokens_per_second']:.2f}")
    print(f"   Tokens/second (after first): {stats['tokens_per_second_after_first']:.2f}")
    
    return {
        'init_time': init_time,
        'forward_time': forward_time,
        'gen_time': gen_time,
        'first_token_time': stats['first_token_time'],
        'tokens_per_second': stats['tokens_per_second'],
    }


def demo_sharding():
    """Demo sharding utilities."""
    print("\n" + "="*80)
    print("SHARDING DEMO")
    print("="*80)
    
    # Check available devices
    devices = jax.devices()
    print(f"\nAvailable devices: {len(devices)}")
    
    # Create mesh
    num_devices = min(4, len(devices))
    mesh = create_device_mesh(
        num_devices=num_devices,
        mesh_shape=(num_devices,),
        axis_names=('data',)
    )
    
    print(f"\nMesh configuration:")
    print(f"  Shape: {mesh.devices.shape}")
    print(f"  Axes: {mesh.axis_names}")
    
    # Create sample data
    batch = jnp.ones((8, 16, 64))
    print(f"\nBatch shape: {batch.shape}")
    
    # Shard data
    batch_spec, _ = get_data_parallel_sharding()
    sharding = jax.sharding.NamedSharding(mesh, batch_spec)
    
    with mesh:
        sharded_batch = jax.device_put(batch, sharding)
    
    print_sharding_info(sharded_batch, "Sharded batch")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("JAX OPTIMIZATION DEMO")
    print("="*80)
    print("\nThis demo shows the impact of different JAX optimizations")
    print("for the gpt-oss-20b model.")
    
    # Demo 1: Baseline
    baseline_config = demo_baseline_config()
    baseline_stats = demo_generation(baseline_config, "Baseline (no optimizations)")
    
    # Demo 2: Optimized
    optimized_config = demo_optimized_config()
    optimized_stats = demo_generation(optimized_config, "Optimized (GQA + MoE)")
    
    # Demo 3: Sharding
    demo_sharding()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nPerformance comparison:")
    print(f"  Init time:          {baseline_stats['init_time']:.2f}s -> {optimized_stats['init_time']:.2f}s")
    print(f"  Forward pass:       {baseline_stats['forward_time']*1000:.2f}ms -> {optimized_stats['forward_time']*1000:.2f}ms")
    print(f"  Generation time:    {baseline_stats['gen_time']:.2f}s -> {optimized_stats['gen_time']:.2f}s")
    print(f"  Tokens/second:      {baseline_stats['tokens_per_second']:.2f} -> {optimized_stats['tokens_per_second']:.2f}")
    
    speedup = baseline_stats['tokens_per_second'] / optimized_stats['tokens_per_second']
    if speedup > 1:
        print(f"\n  Speedup: {speedup:.2f}x faster with optimizations")
    else:
        print(f"\n  Note: Optimizations may have minimal impact on small models/CPU")
        print(f"        For best results, test on TPU/GPU with larger models")
    
    print("\n" + "="*80)
    print("Demo complete!")
    print("\nFor more details, see OPTIMIZATION_GUIDE.md")


if __name__ == "__main__":
    main()
