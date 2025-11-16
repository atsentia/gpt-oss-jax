# JAX Optimization Guide for GPT-OSS-20B

This guide documents the JAX optimizations implemented in this codebase, inspired by the [Lightricks blog post](https://cloud.google.com/blog/products/media-entertainment/how-lightricks-trains-video-diffusion-models-at-scale-with-jax-on-tpu/) on scaling JAX training on TPUs.

## Overview

The gpt-oss-20b JAX implementation includes several optimization strategies to improve performance on CPUs, GPUs, and TPUs:

1. **Grouped-Query Attention (GQA) Broadcasting** - Memory-efficient attention computation
2. **MoE Token Grouping** - Better cache locality for Mixture of Experts
3. **KV Caching** - Efficient autoregressive generation
4. **JIT Compilation** - XLA optimization for compute kernels
5. **Precision Management** - BF16/FP8 for memory and compute efficiency

## Optimization Flags

### 1. GQA Broadcasting Optimization (`optimize_gqa`)

**What it does**: Instead of explicitly expanding Key and Value tensors to match Query dimensions (memory copy), uses broadcasting to achieve the same result.

**Benefits**:
- Reduces memory usage during attention computation
- Eliminates redundant data copies
- Particularly effective with large KV cache

**Usage**:
```python
config = ModelConfig(
    optimize_gqa=True,  # Enable GQA broadcasting
    num_attention_heads=64,
    num_key_value_heads=8,  # 8x reduction in KV heads
    # ... other config
)
```

**Implementation**: See `sdpa()` function in `model.py`:
- **Baseline**: `K = K[:, :, None, :].repeat(q_mult, axis=2)` - explicit copy
- **Optimized**: `K = K[:, :, None, :]` - uses broadcasting, no copy

**Expected Impact**:
- Memory: ~8x reduction in KV tensor size during attention (for q_mult=8)
- Speed: Slightly faster due to less memory traffic

### 2. MoE Token Grouping (`optimize_moe`)

**What it does**: Groups tokens by their assigned experts and processes them together, improving cache locality.

**Benefits**:
- Better L1/L2/L3 cache utilization
- Reduces expert weight loading overhead
- More efficient on TPU with high memory bandwidth

**Usage**:
```python
config = ModelConfig(
    optimize_moe=True,  # Enable MoE token grouping
    num_experts=128,
    experts_per_token=4,
    # ... other config
)
```

**Implementation**: See `MLPBlock.__call__()` in `model.py`:
- **Baseline**: Process each token independently, load expert weights per token
- **Optimized**: Build assignment matrix, process all tokens for each expert together

**Expected Impact**:
- Speed: 10-30% faster MLP computation (depends on expert distribution)
- Memory: Slightly higher temporary memory for assignment matrix

### 3. KV Caching (`use_kv_cache`)

**What it does**: Caches Key and Value tensors from previous tokens to avoid recomputation during autoregressive generation.

**Benefits**:
- O(1) instead of O(n²) compute per new token
- Essential for interactive generation
- 50-100x speedup for long sequences

**Usage**:
```python
from gpt_oss.jax.inference import generate

tokens = generate(
    model=model,
    params=params,
    prompt_tokens=prompt,
    use_kv_cache=True,  # Enable KV caching (default)
    config=config,
    max_new_tokens=100
)
```

**Implementation**: See `KVCache` class in `kv_cache.py`
- Maintains sliding window cache per layer
- Automatically handles cache invalidation and updates
- Supports batched generation

**Expected Impact**:
- First token: No change (cache initialization)
- Subsequent tokens: 50-100x faster

### 4. JIT Compilation (`jit_generate_loop`)

**What it does**: Uses JAX's JIT compiler to optimize the generation loop.

**Benefits**:
- XLA optimization of compute kernels
- Fusion of operations
- Better instruction scheduling

**Usage**:
```python
tokens = generate(
    model=model,
    params=params,
    prompt_tokens=prompt,
    jit_generate_loop=True,  # Enable JIT compilation
    config=config,
    max_new_tokens=100
)
```

**Limitations**:
- First compilation is slow (warmup)
- CPU: Minimal benefit (5-10% speedup)
- TPU/GPU: Significant benefit (2-3x speedup)

**Expected Impact**:
- CPU: 5-10% speedup after warmup
- TPU: 2-3x speedup after warmup
- GPU: 1.5-2x speedup after warmup

## Precision Management

### BF16 (Brain Float 16)

**Default precision** for the model:
- 16-bit floating point optimized for ML workloads
- TPU v2-8: ~42GB memory for 20B model
- Good balance of speed and accuracy

### FP8 (8-bit Floating Point)

**Experimental** quantization for inference:
- 8-bit floating point for extreme memory savings
- TPU v6e: ~21GB memory for 20B model
- Slight accuracy loss but acceptable for many tasks

**Usage**: See `quantization/` module for FP8 conversion utilities.

## Performance Benchmarking

Use the provided benchmark script to measure optimization impact:

```bash
python scripts/benchmark_optimizations.py --config baseline
python scripts/benchmark_optimizations.py --config optimized
```

Expected results on different hardware:

| Hardware | Baseline | +GQA | +MoE | +KV | +JIT | Total Speedup |
|----------|----------|------|------|-----|------|---------------|
| CPU (M1) | 1.0x | 1.05x | 1.15x | 50x | 1.1x | ~60x |
| TPU v2-8 | 1.0x | 1.1x | 1.3x | 50x | 2.5x | ~180x |
| A100 GPU | 1.0x | 1.08x | 1.2x | 50x | 2.0x | ~130x |

*Note: KV cache speedup applies only to subsequent tokens, not the first token*

## Best Practices

### For CPU Development

```python
config = ModelConfig(
    optimize_gqa=True,   # Memory efficient
    optimize_moe=False,  # Minimal benefit on CPU
    use_scan_layers=False,  # Not yet implemented
)

# JIT provides minimal benefit on CPU
tokens = generate(..., jit_generate_loop=False, use_kv_cache=True)
```

### For TPU Production

```python
config = ModelConfig(
    optimize_gqa=True,   # Essential for memory
    optimize_moe=True,   # Significant speedup on TPU
    use_scan_layers=False,  # Future optimization
)

# JIT essential for TPU performance
tokens = generate(..., jit_generate_loop=True, use_kv_cache=True)
```

### For GPU Inference

```python
config = ModelConfig(
    optimize_gqa=True,   # Memory efficient
    optimize_moe=True,   # Good speedup on GPU
    use_scan_layers=False,  # Future optimization
)

# JIT recommended for GPU
tokens = generate(..., jit_generate_loop=True, use_kv_cache=True)
```

## Future Optimizations

### 1. lax.scan for Transformer Layers (Planned)

Replace Python loop with `nn.scan` for faster compilation:
- Currently blocked by complexity with KV caching
- Would provide 10-20% faster compilation
- Lower memory usage during tracing

### 2. Flash Attention (Partial)

Memory-efficient attention using tiling:
- Code stub exists but not fully integrated
- Would enable longer sequences on limited memory
- 2-4x memory reduction for attention

### 3. Pipeline Parallelism

Split model across multiple devices:
- Essential for >70B models
- Requires sharding annotations
- Planned for future release

### 4. Gradient Checkpointing (Training)

Trade compute for memory during backprop:
- Not applicable for inference-only
- Would enable larger batch sizes for training
- Planned if training is added

## Troubleshooting

### Out of Memory (OOM)

1. Enable all optimizations: `optimize_gqa=True`, `optimize_moe=True`
2. Use KV cache: `use_kv_cache=True`
3. Consider FP8 quantization (experimental)
4. Reduce batch size or sequence length

### Slow Compilation

1. Disable JIT during development: `jit_generate_loop=False`
2. Use smaller model for testing
3. Compilation is one-time cost - subsequent runs are fast

### Accuracy Issues

1. Verify FP8 quantization is not enabled
2. Check that model weights are loaded correctly
3. Compare with PyTorch reference implementation
4. Disable optimizations one by one to isolate issue

## References

- [Lightricks: Training Video Diffusion Models at Scale with JAX on TPU](https://cloud.google.com/blog/products/media-entertainment/how-lightricks-trains-video-diffusion-models-at-scale-with-jax-on-tpu/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [Orbax Documentation](https://orbax.readthedocs.io/)
