# Colab TPU Notebook Plan: `jax_inference_colab_tpu.ipynb`

## Overview

Adaptive precision inference notebook for Google Colab TPUs with:
- **TPU v2-8**: MXFP4 → BF16 (16-bit), ~42GB memory
- **TPU v6e**: MXFP4 → FP8 (8-bit), ~21GB memory
- Automatic TPU detection and precision selection
- HuggingFace model download
- Orbax conversion with adaptive precision
- TPU utilization monitoring

---

## Cell Structure (15 cells)

### Cell 1: Title & Overview (Markdown)

```markdown
# GPT-OSS-20B on Google Colab TPU
## Adaptive Precision Inference

This notebook demonstrates **production-grade adaptive inference** that automatically selects optimal precision based on TPU hardware:

| TPU Type | Memory | Strategy | Model Size |
|----------|--------|----------|------------|
| **v2-8** | 64GB (8x8GB) | BF16 (16-bit) | ~42GB |
| **v6e** | 32GB | FP8 (8-bit) | ~21GB |

### Features
- 🎯 Automatic TPU detection and precision selection
- 📦 HuggingFace model download (13.8GB)
- ⚡ Fast Orbax checkpoint format (5s load time)
- 🎨 Harmony protocol with multi-channel reasoning
- 📊 TPU utilization monitoring

### Requirements
- **Runtime**: TPU (v2-8 or v6e) - Select in Colab: Runtime → Change runtime type → TPU
- **Time**: ~15 minutes first run (download + conversion)
- **Storage**: ~30GB temporary (cleaned up automatically)
```

---

### Cell 2: Environment Setup (Code)

```python
# Install JAX with TPU support
!pip install -q "jax[tpu]>=0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install -q flax>=0.8.0 orbax-checkpoint>=0.5.0 safetensors>=0.5.3
!pip install -q openai-harmony tiktoken>=0.9.0 tqdm huggingface_hub

# Clone repository
!git clone -q https://github.com/atsentia/gpt-oss-jax.git 2>/dev/null || echo "Repo already cloned"
%cd gpt-oss-jax

# Install package
!pip install -q -e ".[jax]"

print("✓ Environment setup complete")
```

---

### Cell 3: TPU Detection & Precision Selection (Code) ⭐

```python
import jax
import jax.numpy as jnp

# Detect TPU
devices = jax.devices()
backend = jax.default_backend()

if backend != "tpu":
    raise RuntimeError(
        f"❌ TPU not detected (found: {backend})\n"
        "Please change runtime: Runtime → Change runtime type → TPU"
    )

tpu_type = devices[0].device_kind
num_devices = len(devices)

print(f"✓ Detected: {tpu_type}")
print(f"  Devices: {num_devices}")

# Determine precision strategy
if "v2" in tpu_type and num_devices == 8:
    STRATEGY = "bf16"
    TARGET_DTYPE = jnp.bfloat16
    EXPECTED_MEMORY_GB = 42
    CONVERSION_DTYPE = "bf16"
    print(f"\n📊 Strategy: BF16 (16-bit)")
    print(f"   Memory: ~{EXPECTED_MEMORY_GB}GB (fits in 64GB HBM)")
    print(f"   TPU v2-8 has ample memory for full precision")

elif "v6" in tpu_type:
    STRATEGY = "fp8"
    TARGET_DTYPE = jnp.float8_e4m3fn
    EXPECTED_MEMORY_GB = 21
    CONVERSION_DTYPE = "fp8_e4m3fn"
    print(f"\n📊 Strategy: FP8 (8-bit)")
    print(f"   Memory: ~{EXPECTED_MEMORY_GB}GB (fits in 32GB HBM)")
    print(f"   TPU v6e requires quantization for 20B model")

elif "v5" in tpu_type:
    print(f"\n⚠️  TPU v5e detected")
    print(f"   Estimated HBM: 16-24GB (insufficient for GPT-OSS-20B)")
    print(f"   Even with FP8 (21GB), this may not fit")
    print(f"   Consider using smaller model or TPU v6e")
    raise RuntimeError("Insufficient memory for GPT-OSS-20B")

else:
    raise RuntimeError(f"Unsupported TPU type: {tpu_type}")

print(f"\n✓ Precision strategy: {STRATEGY.upper()}")
```

---

### Cell 4: Download from HuggingFace (Code)

```python
from huggingface_hub import snapshot_download
from pathlib import Path

# Download GPT-OSS-20B SafeTensors checkpoint
print("Downloading GPT-OSS-20B from HuggingFace...")
print("Size: 13.8 GB (may take 5-10 minutes)")

checkpoint_dir = snapshot_download(
    repo_id="openai/gpt-oss-20b",
    revision="main",
    allow_patterns=["original/*"],
    local_dir="/content/gpt-oss-20b-download",
    local_dir_use_symlinks=False,
)

safetensors_path = Path("/content/gpt-oss-20b-download/original")

# Verify download
assert safetensors_path.exists(), "Download failed"
assert (safetensors_path / "model.safetensors").exists(), "Missing model file"

print(f"✓ Downloaded to: {safetensors_path}")
print(f"  Files:")
for f in safetensors_path.glob("*"):
    size_mb = f.stat().st_size / 1e6
    print(f"    - {f.name}: {size_mb:.1f} MB")
```

---

### Cell 5: Load SafeTensors with Adaptive Precision (Code) ⭐

```python
import time
from gpt_oss.jax.config import ModelConfig
from gpt_oss.jax.loader_safetensors import WeightLoader
from gpt_oss.jax.quantization import decompress_mxfp4_to_dtype

# Load config
config = ModelConfig()  # Default GPT-OSS-20B config

# Load weights with adaptive precision
print(f"Loading checkpoint with {STRATEGY.upper()} precision...")
print(f"This will decompress MXFP4 (4-bit) → {STRATEGY.upper()} ({EXPECTED_MEMORY_GB}GB)")

load_start = time.time()

# Load SafeTensors (decompresses MXFP4 to target dtype)
loader = WeightLoader(str(safetensors_path))

# For BF16: use default (existing code)
# For FP8: we'll convert after loading
params = loader.load_params(config, show_progress=True)

load_time = time.time() - load_start

print(f"\n✓ Loaded in {load_time:.1f}s")

# Convert to FP8 if needed
if STRATEGY == "fp8":
    print(f"\nConverting BF16 → FP8...")
    params = jax.tree_util.tree_map(
        lambda x: x.astype(TARGET_DTYPE) if x.dtype == jnp.bfloat16 else x,
        params
    )
    print(f"✓ Converted to FP8")
```

---

### Cell 6: Save to Orbax Format (Code)

```python
from orbax.checkpoint import PyTreeCheckpointer
import orbax.checkpoint as ocp

# Save to Orbax for fast loading in future runs
orbax_path = Path(f"/content/gpt-oss-20b-orbax-{STRATEGY}")

print(f"Saving Orbax checkpoint ({STRATEGY.upper()})...")
print(f"Path: {orbax_path}")

checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save(
    orbax_path,
    params,
    save_args=ocp.SaveArgs(aggregate=True)
)

print(f"✓ Saved Orbax checkpoint")
print(f"  Future runs will load in ~5s instead of {load_time:.0f}s!")
```

---

### Cell 7: Model Initialization (Code)

```python
from gpt_oss.jax.model import Transformer
from gpt_oss.tokenizer import get_tokenizer

# Initialize model
model = Transformer(config=config)
tokenizer = get_tokenizer()

# Verify dtype
sample_weight = jax.tree_util.tree_leaves(params)[0]

print(f"✓ Model initialized")
print(f"  Architecture: GPT-OSS-20B")
print(f"  Layers: {config.num_hidden_layers}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Attention heads: {config.num_attention_heads}")
print(f"  MoE experts: {config.num_experts}")
print(f"\n  Weight dtype: {sample_weight.dtype}")
print(f"  Precision: {STRATEGY.upper()}")
```

---

### Cell 8: Memory Estimation (Code) ⭐

```python
def estimate_memory(params):
    """Estimate memory usage in GB."""
    total_bytes = sum(p.nbytes for p in jax.tree_util.tree_leaves(params))
    return total_bytes / 1e9

actual_memory_gb = estimate_memory(params)

print(f"📊 Memory Usage:")
print(f"  Actual: {actual_memory_gb:.2f} GB")
print(f"  Expected: {EXPECTED_MEMORY_GB} GB")
print(f"  Utilization: {(actual_memory_gb/EXPECTED_MEMORY_GB)*100:.1f}%")

# TPU HBM capacity
tpu_hbm_gb = 64 if "v2" in tpu_type and num_devices == 8 else 32
print(f"\n  TPU HBM Capacity: {tpu_hbm_gb} GB")
print(f"  Available: {tpu_hbm_gb - actual_memory_gb:.1f} GB")

if actual_memory_gb > tpu_hbm_gb * 0.9:
    print(f"\n⚠️  Warning: Using >90% of TPU memory!")
    print(f"   May encounter OOM errors during inference")
elif actual_memory_gb <= tpu_hbm_gb * 0.7:
    print(f"\n✓ Comfortable fit ({(actual_memory_gb/tpu_hbm_gb)*100:.0f}% of HBM)")
```

---

### Cell 9: Harmony Protocol Demo (Code) ⭐

```python
import re
from IPython.display import HTML, display
from gpt_oss.jax.inference import generate

# Try to use openai-harmony for Harmony formatting
try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        Conversation,
        Message,
        Role
    )

    user_message = "What is the capital of France?"

    print("=" * 70)
    print("Harmony Format Example - Multi-Channel Reasoning")
    print("=" * 70)

    # Display original message (blue box)
    display(HTML(f"""
    <div style="margin: 15px 0; padding: 15px; background-color: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px;">
        <h3 style="margin-top: 0; color: #1976d2;">a) Original User Message</h3>
        <p style="font-size: 16px; font-family: monospace; color: #0d47a1; margin: 0;">
            {user_message}
        </p>
    </div>
    """))

    # Load Harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.USER, user_message)
    ])
    prompt_tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    # Display Harmony prompt (green box)
    prompt_text_escaped = tokenizer.decode(prompt_tokens).replace('<', '&lt;').replace('>', '&gt;')
    display(HTML(f"""
    <div style="margin: 15px 0; padding: 15px; background-color: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 4px;">
        <h3 style="margin-top: 0; color: #2e7d32;">b) Harmony-Formatted Prompt</h3>
        <p style="font-size: 14px; font-family: monospace; color: #1b5e20; margin: 0; white-space: pre-wrap;">
            {prompt_text_escaped}
        </p>
        <p style="font-size: 12px; color: #558b2f; margin-top: 10px;">
            ✓ Tokens: {len(prompt_tokens)} | Special tokens: {[t for t in prompt_tokens if t >= 199998]}
        </p>
    </div>
    """))

    # Generate
    print("\nGenerating response...")
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    output_tokens, stats = generate(
        model=model,
        params=params,
        prompt_tokens=prompt_tokens,
        max_new_tokens=50,
        temperature=0.0,
        rng_key=jax.random.PRNGKey(42),
        config=config,
        use_kv_cache=True,
        show_progress=False,
        return_stats=True
    )

    # Filter stop tokens
    filtered_tokens = [t for t in output_tokens[len(prompt_tokens):] if t not in stop_token_ids]
    generated_only = tokenizer.decode(filtered_tokens)
    generated_escaped = generated_only.replace('<', '&lt;').replace('>', '&gt;')

    # Display raw response (yellow box)
    display(HTML(f"""
    <div style="margin: 15px 0; padding: 15px; background-color: #fff9c4; border-left: 4px solid #fbc02d; border-radius: 4px;">
        <h3 style="margin-top: 0; color: #f57f17;">c) Full Generated Response (raw Harmony format)</h3>
        <p style="font-size: 14px; font-family: monospace; color: #827717; margin: 0; white-space: pre-wrap;">
            {generated_escaped}
        </p>
    </div>
    """))

    # Parse channels
    reasoning_match = re.search(r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)', generated_only, re.DOTALL)
    answer_match = re.search(r'<\|channel\|>(main|final)<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)', generated_only, re.DOTALL)

    # Display reasoning (green box)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        display(HTML(f"""
        <div style="margin: 15px 0; padding: 15px; background-color: #c8e6c9; border-left: 4px solid #66bb6a; border-radius: 4px;">
            <h4 style="margin-top: 0; color: #2e7d32;">📊 Reasoning/Analysis (analysis channel)</h4>
            <div style="font-size: 15px; color: #1b5e20;">{reasoning}</div>
        </div>
        """))

    # Display final answer (purple box)
    if answer_match:
        answer = answer_match.group(2).strip()
        display(HTML(f"""
        <div style="margin: 15px 0; padding: 15px; background-color: #f3e5f5; border-left: 4px solid #ab47bc; border-radius: 4px;">
            <h4 style="margin-top: 0; color: #7b1fa2;">💬 Final Answer (final channel)</h4>
            <div style="font-size: 15px; color: #4a148c;">{answer}</div>
        </div>
        """))

    # Performance stats
    print(f"\n📊 Performance: {stats['total_time']:.2f}s | {stats['tokens_per_second']:.2f} tok/s")

except ImportError:
    print("⚠️  openai-harmony not installed")
    print("   Run: !pip install openai-harmony")
```

---

### Cell 10: Performance Comparison Table (Markdown) ⭐

```markdown
## Performance Comparison: Adaptive Precision

| Metric | TPU v2-8 (BF16) | TPU v6e (FP8) |
|--------|-----------------|---------------|
| **Precision** | 16-bit | 8-bit |
| **Model Memory** | ~42 GB | ~21 GB |
| **TPU HBM** | 64 GB (8x8GB) | 32 GB |
| **Memory Utilization** | 66% | 66% |
| **Load Time** | ~5s (Orbax) | ~5s (Orbax) |
| **Tokens/sec (est.)** | 50-100 | 80-150* |
| **Cost/hour** | $2.40 | $1.60** |

\* FP8 may be faster due to lower memory bandwidth requirements
\** Estimated - check current Colab pricing

### Key Insights

1. **2x Memory Reduction**: FP8 uses half the memory of BF16
2. **Same Load Speed**: Orbax provides fast loading regardless of precision
3. **Potential Speedup**: FP8 inference may be faster on memory-bound workloads
4. **Cost Efficiency**: Smaller TPUs become viable with quantization
5. **Quality**: MXFP4→FP8 maintains model quality (designed for 8-bit)
```

---

### Cell 11: Save to Google Drive (Code - Optional)

```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Save Orbax checkpoint to Drive
drive_path = f"/content/drive/MyDrive/gpt-oss-20b-orbax-{STRATEGY}"

print(f"Saving checkpoint to Google Drive...")
print(f"Path: {drive_path}")
print(f"Size: ~{EXPECTED_MEMORY_GB}GB")
print(f"This may take 10-15 minutes...")

!cp -r {orbax_path} "{drive_path}"

print(f"✓ Saved to Google Drive")
print(f"  Reusable across Colab sessions!")
print(f"  Load with: OrbaxWeightLoader('{drive_path}')")
```

---

### Cell 12: TPU Utilization Monitoring (Code) ⭐

```python
import subprocess
import json

print("=" * 70)
print("TPU Utilization Monitoring")
print("=" * 70)

# Method 1: JAX device memory stats (if available)
try:
    from jax.lib import xla_bridge
    backend = xla_bridge.get_backend()

    print("\n📊 JAX Device Stats:")
    for i, device in enumerate(devices):
        print(f"\n  Device {i}: {device}")
        try:
            mem_info = backend.get_memory_info(device)
            if mem_info:
                used_gb = mem_info.bytes_in_use / 1e9
                limit_gb = mem_info.bytes_limit / 1e9
                print(f"    Memory: {used_gb:.2f} / {limit_gb:.2f} GB ({used_gb/limit_gb*100:.1f}%)")
        except:
            print(f"    Memory info not available")

except Exception as e:
    print(f"⚠️  JAX device stats not available: {e}")

# Method 2: Cloud TPU monitoring (if running on GCP)
try:
    print("\n\n📊 Cloud TPU Metrics:")

    # Try to get TPU name from metadata
    result = subprocess.run(
        ["curl", "-H", "Metadata-Flavor: Google",
         "http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-name"],
        capture_output=True,
        text=True,
        timeout=2
    )

    if result.returncode == 0:
        tpu_name = result.stdout.strip()
        print(f"  TPU Name: {tpu_name}")
        print(f"  Use 'gcloud' or Cloud Console to view detailed metrics")
    else:
        print(f"  Running on Colab (TPU metrics via Cloud Console)")

except:
    print(f"  Cloud TPU monitoring not available in Colab")

# Method 3: System-level monitoring
print("\n\n💻 System Info:")
!nvidia-smi 2>/dev/null || echo "  No NVIDIA GPU (expected on TPU runtime)"
!cat /proc/meminfo | grep MemTotal || echo "  System memory info"
!df -h /content | tail -1

print("\n\n📝 TPU Monitoring Tips:")
print("  1. Colab → Runtime → View runtime logs")
print("  2. Check for OOM errors in logs")
print("  3. Monitor generation speed (tok/s) as proxy for utilization")
print("  4. For detailed metrics: Use GCP Console (if on Cloud TPU)")
```

---

### Cell 13: Cleanup (Code - Optional)

```python
# Free up Colab storage
print("Cleaning up temporary files...")

!rm -rf /content/gpt-oss-20b-download
!rm -rf /content/gpt-oss-20b-orbax-*  # Keep if saved to Drive

print("✓ Cleanup complete")
print(f"\nReclaimed space:")
!df -h /content | tail -1
```

---

### Cell 14: Troubleshooting (Markdown)

```markdown
## Troubleshooting

### OOM (Out of Memory) Errors

**Symptom**: `RESOURCE_EXHAUSTED` or crash during loading/inference

**Solutions**:
1. Verify TPU type matches strategy:
   - TPU v2-8: Use BF16 (Cell 3 should select this)
   - TPU v6e: Use FP8 (Cell 3 should select this)
   - TPU v5e: Not supported (insufficient memory)

2. Restart runtime and re-run from Cell 1

3. Check memory usage in Cell 8 - should be <70% of HBM

### Slow Download

**Symptom**: Cell 4 takes >15 minutes

**Solutions**:
- HuggingFace may be rate-limiting
- Wait and retry
- Alternative: Download to Google Drive first, then copy to /content/

### TPU Not Detected

**Symptom**: Cell 3 shows `backend: cpu` or `backend: cuda`

**Solutions**:
1. Runtime → Change runtime type → TPU v2 or TPU
2. Restart runtime
3. Re-run Cell 2 and 3

### Import Errors

**Symptom**: `ModuleNotFoundError` in Cell 2 or later

**Solutions**:
1. Re-run Cell 2 (environment setup)
2. Check for error messages during pip install
3. Restart runtime if persistent

### Harmony Parsing Fails

**Symptom**: Cell 9 shows raw tokens instead of channels

**Solutions**:
- Model may need fine-tuning for Harmony format
- Check `openai-harmony` version: `!pip show openai-harmony`
- Expected: Model trained with Harmony should output proper channels
```

---

### Cell 15: Optimization Exercises (Markdown) ⭐

```markdown
## 🚀 Optimization Exercises: Take It Further!

Now that you've run GPT-OSS-20B on TPU, here are optimization opportunities to explore:

### 1. JAX Code Optimization

**Current state**: Standard JAX implementation with XLA compilation

**Potential improvements**:
- **Fused kernels**: Combine LayerNorm + Attention into single kernel
- **Scan loops**: Replace Python loops with `jax.lax.scan` for better compilation
- **Sharding**: Distribute model across multiple TPUs with `jax.sharding`
- **Mixed precision training**: FP8 activations during fine-tuning

**Learn more**:
- [JAX Performance Tips](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [Our code](https://github.com/atsentia/gpt-oss-jax/blob/main/gpt_oss/jax/model.py) - See `Transformer` class

**Exercise**: Profile inference with `jax.profiler` and identify bottlenecks

---

### 2. KV Cache Optimization

**Current state**: Static pre-allocated cache (4096 tokens)

**Potential improvements**:
- **Dynamic caching**: Grow cache on-demand instead of pre-allocating
- **Paged attention**: Split cache into blocks like vLLM
- **Quantized cache**: Store K/V in INT8 or FP8 (2-4x memory savings)
- **Ring attention**: Enable infinite context via ring buffer

**Memory impact**:
- Current: `batch_size × 4096 × num_layers × kv_heads × head_dim × 2 bytes(BF16)`
- FP8 cache: 50% reduction
- Paged: 70-80% reduction (no pre-allocation)

**Learn more**:
- [vLLM PagedAttention](https://github.com/vllm-project/vllm)
- [Our KV cache](https://github.com/atsentia/gpt-oss-jax/blob/main/gpt_oss/jax/kv_cache.py)

**Exercise**: Implement INT8 KV cache quantization

---

### 3. Advanced Quantization

**Current state**: MXFP4 (4-bit) → FP8/BF16 decompression at load time

**Potential improvements**:
- **On-the-fly dequantization**: Keep weights in MXFP4, dequantize per-layer
  - Memory: 10.5 GB instead of 21/42 GB
  - Tradeoff: +10-20% compute overhead
- **Activation quantization**: Quantize intermediate activations to INT8
  - Reduces memory bandwidth requirements
  - Faster on TPU (INT8 ops are highly optimized)
- **Per-channel quantization**: Separate scale per output channel
  - Better accuracy than per-tensor
- **GPTQ/AWQ**: Post-training quantization methods
  - May achieve 3-bit with minimal accuracy loss

**Learn more**:
- [Our quantization code](https://github.com/atsentia/gpt-oss-jax/tree/main/gpt_oss/jax/quantization)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)

**Exercise**: Benchmark MXFP4 on-the-fly vs pre-decompressed

---

### 4. Speculative Decoding

**Current state**: Autoregressive generation (one token at a time)

**Potential improvements**:
- **Draft model**: Use smaller model (e.g., GPT-OSS-1B) to propose tokens
  - Verify 3-5 draft tokens in parallel with main model
  - 2-3x speedup with no accuracy loss
- **Medusa heads**: Train extra prediction heads for multi-token generation
- **Lookahead decoding**: Parallelize token generation within single model

**Speed impact**:
- Current: N tokens = N forward passes
- Speculative (k=4): N tokens ≈ N/2.5 forward passes (1.6x faster)
- Medusa (k=5): N tokens ≈ N/3 forward passes (2x faster)

**Challenges**:
- Need to load draft model (additional memory)
- Token acceptance rate depends on task
- Batch size = 1 (hard to parallelize further)

**Learn more**:
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [Implementation guide](https://github.com/atsentia/gpt-oss-jax/issues) - Create an issue to discuss!

**Exercise**: Implement simple speculative decoding with GPT-2 as draft

---

### 5. Continuous Batching

**Current state**: Single request at a time

**Potential improvements**:
- **Dynamic batching**: Group multiple requests into single batch
  - Amortize model overhead across requests
  - 5-10x higher throughput
- **Request-level KV cache**: Each request has own cache state
  - Enables parallel serving
- **Prefix caching**: Share KV cache for common prompts (e.g., system prompt)
  - 50-80% cache hit rate in production

**Throughput impact**:
- Single request: 50-100 tok/s
- Batch of 8: 300-500 tok/s total (6-10x more efficient)

**Learn more**:
- [Orca continuous batching](https://www.usenix.org/conference/osdi22/presentation/yu)
- [vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview.html)

**Exercise**: Implement simple batched inference (fixed batch size)

---

## 🤔 Your Turn: What Would You Improve?

Given the constraints:
- **TPU v6e**: 32GB HBM
- **Model**: GPT-OSS-20B (21B parameters)
- **Goal**: Maximize throughput (tokens/sec) for production serving

**Discussion questions**:

1. **Memory vs Speed Tradeoff**:
   - FP8 uses 21GB, leaving 11GB for KV cache and activations
   - MXFP4 on-the-fly uses 10.5GB, leaving 21.5GB free
   - Which would you choose and why?

2. **Optimization Priority**:
   - Rank these by expected ROI: Speculative decoding, Quantized KV cache, Continuous batching, Fused kernels
   - What's your reasoning?

3. **Production Deployment**:
   - Single large model (GPT-OSS-20B) or ensemble of smaller models?
   - What workload characteristics favor each approach?

**Share your thoughts**: [Open a discussion on GitHub](https://github.com/atsentia/gpt-oss-jax/discussions)

---

**Next notebook**: Advanced optimizations hands-on (coming soon!)
```

---

### Cell 16: Conclusion & Resources (Markdown)

```markdown
## Conclusion

This notebook demonstrated:

✅ **Adaptive Precision Inference**
- Automatic hardware detection
- Optimal dtype selection (BF16 vs FP8)
- 2x memory reduction with quantization

✅ **Production Patterns**
- HuggingFace model integration
- Fast Orbax checkpointing
- Memory monitoring and optimization

✅ **Harmony Protocol**
- Multi-channel reasoning (analysis + final answer)
- Color-coded visualization
- Structured LLM outputs

### Key Metrics

| Component | Time | Memory |
|-----------|------|--------|
| Download | ~10 min | 14 GB |
| Conversion | ~5 min | - |
| Loading | ~5s | {EXPECTED_MEMORY_GB} GB |
| Inference | Variable | +2-5 GB |

### Next Steps

1. **Fine-tune** for your use case
2. **Deploy** to production TPU instances
3. **Scale** with multi-TPU setups
4. **Optimize** further with Pallas custom kernels

### Resources

- **Repository**: [gpt-oss-jax](https://github.com/atsentia/gpt-oss-jax)
- **Model Card**: [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
- **JAX Documentation**: [jax.readthedocs.io](https://jax.readthedocs.io/)
- **Harmony Protocol**: [github.com/openai/harmony](https://github.com/openai/harmony)
- **Orbax**: [orbax.readthedocs.io](https://orbax.readthedocs.io/)

### Feedback

Found an issue? [Open an issue on GitHub](https://github.com/atsentia/gpt-oss-jax/issues)

---

**🎉 Happy inferencing on TPU!**
```

---

## Summary

**Total cells**: 15
**Estimated runtime**:
- First run: ~15-20 min (download + conversion + inference)
- Subsequent runs: ~2-3 min (if Orbax saved to Drive)

**Key innovations**:
1. Adaptive precision (BF16 vs FP8) based on TPU detection
2. 2x memory reduction for TPU v6e
3. Production-grade patterns (monitoring, error handling)
4. Complete Harmony protocol demo
5. TPU utilization monitoring

**Ready for implementation**: Yes
