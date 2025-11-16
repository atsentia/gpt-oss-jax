#!/usr/bin/env python3
"""Benchmark script for measuring JAX optimization impact.

This script measures the performance of different optimization strategies
for the gpt-oss-20b model on various hardware configurations.

Usage:
    # Baseline (no optimizations)
    python scripts/benchmark_optimizations.py --config baseline
    
    # All optimizations enabled
    python scripts/benchmark_optimizations.py --config optimized
    
    # Custom configuration
    python scripts/benchmark_optimizations.py --optimize-gqa --optimize-moe --use-kv-cache
    
    # Measure compilation time
    python scripts/benchmark_optimizations.py --config optimized --measure-compile-time
"""

import argparse
import time
import json
from typing import Dict, Any

import jax
import jax.numpy as jnp

from gpt_oss.jax.model import Transformer
from gpt_oss.jax.config import ModelConfig
from gpt_oss.jax.inference import generate_greedy


def create_small_test_config(**overrides) -> ModelConfig:
    """Create a small test configuration for benchmarking.
    
    This is a smaller version of gpt-oss-20b to enable faster benchmarking
    on resource-constrained hardware.
    
    Args:
        **overrides: Override default config values
        
    Returns:
        ModelConfig with test parameters
    """
    defaults = {
        'num_hidden_layers': 4,      # Reduced from 24
        'num_experts': 16,            # Reduced from 128
        'experts_per_token': 4,
        'vocab_size': 10000,          # Reduced from 201088
        'hidden_size': 512,           # Reduced from 2880
        'intermediate_size': 1024,    # Reduced from 2880
        'swiglu_limit': 7.0,
        'head_dim': 64,
        'num_attention_heads': 8,     # Reduced from 64
        'num_key_value_heads': 2,     # Reduced from 8
        'sliding_window': 64,         # Reduced from 128
        'initial_context_length': 2048,  # Reduced from 4096
        'rope_theta': 150000.0,
        'rope_scaling_factor': 1.0,   # No scaling for test
        'rope_ntk_alpha': 1.0,
        'rope_ntk_beta': 32.0,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def benchmark_forward_pass(
    model: Transformer,
    params: dict,
    num_tokens: int = 32,
    num_iterations: int = 10,
    warmup_iterations: int = 2
) -> Dict[str, float]:
    """Benchmark forward pass latency.
    
    Args:
        model: Transformer model
        params: Model parameters
        num_tokens: Number of tokens in input
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations (excluded from timing)
        
    Returns:
        Dictionary with timing statistics
    """
    # Create random input tokens
    input_tokens = jnp.array([1] * num_tokens, dtype=jnp.int32)
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = model.apply({'params': params}, input_tokens)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        logits = model.apply({'params': params}, input_tokens)
        logits[-1].block_until_ready()  # Wait for computation
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
    }


def benchmark_generation(
    model: Transformer,
    params: dict,
    config: ModelConfig,
    prompt_length: int = 16,
    max_new_tokens: int = 32,
    use_kv_cache: bool = True,
    jit_generate_loop: bool = False
) -> Dict[str, Any]:
    """Benchmark autoregressive generation.
    
    Args:
        model: Transformer model
        params: Model parameters
        config: Model configuration
        prompt_length: Number of tokens in prompt
        max_new_tokens: Number of tokens to generate
        use_kv_cache: Whether to use KV caching
        jit_generate_loop: Whether to use JIT-compiled generation loop
        
    Returns:
        Dictionary with generation statistics
    """
    # Create random prompt
    prompt = [1] * prompt_length
    
    # Time generation
    start = time.time()
    from gpt_oss.jax.inference import generate
    tokens, stats = generate(
        model=model,
        params=params,
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        show_progress=False,
        return_stats=True,
        use_kv_cache=use_kv_cache,
        config=config if use_kv_cache else None,
    )
    total_time = time.time() - start
    
    # Calculate statistics
    first_token_time = stats['token_times'][0] if 'token_times' in stats else 0
    subsequent_times = stats['token_times'][1:] if 'token_times' in stats else []
    
    return {
        'total_time_s': total_time,
        'first_token_time_s': first_token_time,
        'subsequent_token_mean_s': sum(subsequent_times) / len(subsequent_times) if subsequent_times else 0,
        'tokens_per_second': max_new_tokens / total_time if total_time > 0 else 0,
        'tokens_per_second_after_first': (max_new_tokens - 1) / sum(subsequent_times) if subsequent_times else 0,
    }


def benchmark_compilation_time(
    config: ModelConfig,
    num_tokens: int = 32
) -> float:
    """Measure JAX compilation time for the model.
    
    Args:
        config: Model configuration
        num_tokens: Number of tokens for test input
        
    Returns:
        Compilation time in seconds
    """
    # Create model
    model = Transformer(
        config=config,
        optimize_gqa=config.optimize_gqa,
        optimize_moe=config.optimize_moe,
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    input_tokens = jnp.array([1] * num_tokens, dtype=jnp.int32)
    
    # Time initialization (includes compilation)
    start = time.time()
    params = model.init(key, input_tokens)
    compilation_time = time.time() - start
    
    return compilation_time


def run_benchmark_suite(
    config_name: str,
    optimize_gqa: bool,
    optimize_moe: bool,
    use_kv_cache: bool,
    jit_generate_loop: bool,
    measure_compile_time: bool = False
) -> Dict[str, Any]:
    """Run complete benchmark suite.
    
    Args:
        config_name: Name of this configuration
        optimize_gqa: Enable GQA optimization
        optimize_moe: Enable MoE optimization
        use_kv_cache: Enable KV caching
        jit_generate_loop: Enable JIT generation loop
        measure_compile_time: Whether to measure compilation time
        
    Returns:
        Dictionary with all benchmark results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*80}")
    print(f"  optimize_gqa: {optimize_gqa}")
    print(f"  optimize_moe: {optimize_moe}")
    print(f"  use_kv_cache: {use_kv_cache}")
    print(f"  jit_generate_loop: {jit_generate_loop}")
    print()
    
    # Create config
    config = create_small_test_config(
        optimize_gqa=optimize_gqa,
        optimize_moe=optimize_moe,
    )
    
    # Measure compilation time if requested
    compile_time = None
    if measure_compile_time:
        print("Measuring compilation time...")
        compile_time = benchmark_compilation_time(config)
        print(f"  Compilation time: {compile_time:.2f}s")
        print()
    
    # Create model
    model = Transformer(
        config=config,
        optimize_gqa=optimize_gqa,
        optimize_moe=optimize_moe,
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    input_tokens = jnp.array([1] * 32, dtype=jnp.int32)
    params = model.init(key, input_tokens)['params']
    
    # Benchmark forward pass
    print("Benchmarking forward pass (32 tokens, 10 iterations)...")
    forward_stats = benchmark_forward_pass(model, params, num_tokens=32, num_iterations=10)
    print(f"  Mean: {forward_stats['mean_ms']:.2f}ms")
    print(f"  Min: {forward_stats['min_ms']:.2f}ms")
    print(f"  Max: {forward_stats['max_ms']:.2f}ms")
    print(f"  Std: {forward_stats['std_ms']:.2f}ms")
    print()
    
    # Benchmark generation
    print("Benchmarking generation (16 prompt + 32 new tokens)...")
    gen_stats = benchmark_generation(
        model, params, config,
        prompt_length=16,
        max_new_tokens=32,
        use_kv_cache=use_kv_cache,
        jit_generate_loop=jit_generate_loop
    )
    print(f"  Total time: {gen_stats['total_time_s']:.2f}s")
    print(f"  First token: {gen_stats['first_token_time_s']:.2f}s")
    print(f"  Subsequent tokens (mean): {gen_stats['subsequent_token_mean_s']*1000:.2f}ms")
    print(f"  Tokens/second: {gen_stats['tokens_per_second']:.2f}")
    print(f"  Tokens/second (after first): {gen_stats['tokens_per_second_after_first']:.2f}")
    print()
    
    return {
        'config_name': config_name,
        'optimize_gqa': optimize_gqa,
        'optimize_moe': optimize_moe,
        'use_kv_cache': use_kv_cache,
        'jit_generate_loop': jit_generate_loop,
        'compilation_time_s': compile_time,
        'forward_pass': forward_stats,
        'generation': gen_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark JAX optimizations')
    parser.add_argument(
        '--config',
        choices=['baseline', 'optimized', 'custom'],
        default='custom',
        help='Predefined configuration to use'
    )
    parser.add_argument('--optimize-gqa', action='store_true', help='Enable GQA optimization')
    parser.add_argument('--optimize-moe', action='store_true', help='Enable MoE optimization')
    parser.add_argument('--use-kv-cache', action='store_true', default=True, help='Enable KV caching')
    parser.add_argument('--jit-generate-loop', action='store_true', help='Enable JIT generation loop')
    parser.add_argument('--measure-compile-time', action='store_true', help='Measure compilation time')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.config == 'baseline':
        configs = [
            ('Baseline', False, False, False, False),
        ]
    elif args.config == 'optimized':
        configs = [
            ('Optimized (All)', True, True, True, False),
        ]
    else:
        # Custom configuration
        configs = [
            ('Custom', args.optimize_gqa, args.optimize_moe, args.use_kv_cache, args.jit_generate_loop),
        ]
    
    # Run benchmarks
    results = []
    for config_name, opt_gqa, opt_moe, use_kv, jit_gen in configs:
        result = run_benchmark_suite(
            config_name,
            opt_gqa,
            opt_moe,
            use_kv,
            jit_gen,
            args.measure_compile_time
        )
        results.append(result)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for result in results:
        print(f"\n{result['config_name']}:")
        print(f"  Forward pass: {result['forward_pass']['mean_ms']:.2f}ms")
        print(f"  Generation tokens/s: {result['generation']['tokens_per_second']:.2f}")
        print(f"  Generation tokens/s (after first): {result['generation']['tokens_per_second_after_first']:.2f}")
        if result['compilation_time_s'] is not None:
            print(f"  Compilation time: {result['compilation_time_s']:.2f}s")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
