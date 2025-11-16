# JAX Inference Examples

This directory contains example notebooks demonstrating GPT-OSS-20B inference using JAX.

## 📓 Interactive Notebook

**[jax_inference.ipynb](jax_inference.ipynb)** - Complete walkthrough of JAX inference with GPT-OSS-20B

The notebook includes:
- Checkpoint loading (Orbax and SafeTensors formats)
- Model initialization and configuration
- Text generation with temperature sampling
- **Harmony protocol demonstration** with color-coded output showing:
  - Original user prompts
  - Harmony-formatted prompts with special tokens
  - Raw generated responses
  - Parsed reasoning (analysis channel) and final answers (final channel)
- Performance tracking and optimization tips

### Quick Start

```bash
# Install dependencies
uv pip install -e ".[jax,notebook]"

# Setup Jupyter kernel
python -m ipykernel install --user --name=jax-for-gpt-oss

# Launch Jupyter and open jax_inference.ipynb
jupyter notebook jax_inference.ipynb
```

---

## Example Output from Notebook

The sections below show example output from running [jax_inference.ipynb](jax_inference.ipynb):

<div style="background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 5px; padding: 15px; margin: 10px 0; font-family: Arial, sans-serif;">
  <h2 style="color: #856404; margin-top: 0;">
    ⚠️ Performance Notice: CPU-Only Implementation
  </h2>
  <p style="color: #856404; font-size: 14px; line-height: 1.6; margin-bottom: 0;">
    <strong>This notebook runs on CPU only</strong> and is <strong>not yet optimized</strong> for production use.
  </p>
  <ul style="color: #856404; font-size: 14px; line-height: 1.6; margin-top: 10px;">
    <li><strong>No GPU acceleration</strong> - All computation runs on CPU</li>
    <li><strong>No Flash Attention</strong> - Using standard attention implementation</li>
    <li><strong>No speculative decoding</strong> - Standard autoregressive generation</li>
    <li><strong>No quantization optimizations</strong> - Full precision inference</li>
  </ul>
  <p style="color: #856404; font-size: 14px; line-height: 1.6; margin-top: 10px; margin-bottom: 0;">
    Expect <strong>slower generation speeds</strong> compared to optimized GPU implementations.
    This is a reference implementation for development and testing purposes.
  </p>
</div>

## Table of Contents

1. [Device Detection](#device-detection) - Check available compute devices
2. [Checkpoint Loading](#checkpoint-loading) - Load model weights (Orbax or SafeTensors)
3. [Model Initialization](#model-initialization) - Initialize transformer model and tokenizer
4. [Simple Generation](#simple-generation) - Basic text generation example
5. [Streaming Generation with Progress Bar](#streaming-generation-with-progress-bar) - Generation with progress tracking
6. [Temperature Sampling](#temperature-sampling) - Experiment with different temperatures
7. [OpenAI Harmony Prompt Format Example](#openai-harmony-prompt-format-example) - Verify Harmony tokenizer formatting
8. [Conclusion](#conclusion) - Summary and next steps


# JAX Inference with GPT-OSS-20B

This notebook demonstrates how to run inference with the GPT-OSS-20B model using JAX.

## Features

- **Fast checkpoint loading**: Supports both Orbax (5-6s) and SafeTensors formats
- **Efficient generation**: KV caching for autoregressive generation
- **Progress tracking**: Real-time progress bars and performance statistics
- **Temperature sampling**: Configurable randomness for diverse outputs

## Requirements

Install dependencies using one of the methods below:

**Using uv (recommended):**
```bash
uv venv
uv pip install -e ".[jax,notebook]"
```

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[jax,notebook]"
```

**Setup Jupyter kernel:**
```bash
source .venv/bin/activate
python -m ipykernel install --user --name=jax-for-gpt-oss --display-name "Python (jax-for-gpt-oss)"
```

For detailed installation instructions, see [INSTALL.md](../../INSTALL.md).

**Note**: Update the `CHECKPOINT_PATH` variable below to point to your checkpoint directory.


```python
# Standard imports
import jax
import jax.numpy as jnp
from pathlib import Path
import time
from tqdm.auto import tqdm

# GPT-OSS JAX imports
from gpt_oss.jax.config import ModelConfig
from gpt_oss.jax.model import Transformer
from gpt_oss.jax.inference import generate, sample_token
from gpt_oss.jax.loader_orbax import OrbaxWeightLoader, load_config_from_orbax
from gpt_oss.jax.loader_safetensors import WeightLoader
from gpt_oss.tokenizer import get_tokenizer

print("✓ All imports successful")
```

    ✓ All imports successful


import os

# XLA compiler flags for CPU optimization
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=true --xla_cpu_enable_fast_min_max=true"

# Compilation cache directory (speeds up subsequent runs)
cache_dir = Path.home() / ".cache" / "jax"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

# Optional: Enable compilation logging (useful for debugging)
# os.environ["JAX_LOG_COMPILES"] = "1"

print(f"✓ XLA flags configured")
print(f"✓ Compilation cache: ~/.cache/jax")



```python
import os

# XLA compiler flags for CPU optimization
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=true --xla_cpu_enable_fast_min_max=true"

# Compilation cache directory (speeds up subsequent runs)
print(f"✓ Compilation cache: ~/.cache/jax")
print(f"✓ Compilation cache: ~/.cache/jax")
print(f"✓ Compilation cache: ~/.cache/jax")

# Optional: Enable compilation logging (useful for debugging)
# os.environ["JAX_LOG_COMPILES"] = "1"

print(f"✓ XLA flags configured")
print(f"✓ Compilation cache: ~/.cache/jax")

```

    ✓ Compilation cache: ~/.cache/jax
    ✓ Compilation cache: ~/.cache/jax
    ✓ Compilation cache: ~/.cache/jax
    ✓ XLA flags configured
    ✓ Compilation cache: ~/.cache/jax


## Device Detection

Check which devices JAX is using for computation. On CPU-only systems, this will show CPU devices.


```python
# Get available devices
devices = jax.devices()
backend = jax.default_backend()

print(f"Backend: {backend}")
print(f"Devices: {len(devices)}")
for i, device in enumerate(devices):
    print(f"  Device {i}: {device.device_kind} (ID: {device.id})")
```

    Backend: cpu
    Devices: 1
      Device 0: cpu (ID: 0)


## Checkpoint Loading

The notebook automatically detects whether you're using an Orbax or SafeTensors checkpoint:

- **Orbax**: Pre-converted format, loads in ~5-6 seconds
- **SafeTensors**: Original format, includes MXFP4 decompression, loads in ~90 seconds

### Setup Checkpoint Path

**Recommended**: Create a symlink in the `weights/` directory:

```bash
# From the repository root
ln -s /path/to/your/checkpoint weights/gpt-oss-20b
```

Then use the relative path in the notebook: `../weights/gpt-oss-20b`

**Alternative**: Update `CHECKPOINT_PATH` in the cell below to point directly to your checkpoint directory.


```python
# Suppress Orbax warnings about sharding info (prevents local path exposure)
import warnings
warnings.filterwarnings("ignore", message=".*Sharding info not provided.*")
warnings.filterwarnings("ignore", category=UserWarning, module="orbax.*")
# Suppress asyncio errors from Jupyter/IPython kernel
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio.*")
import logging
logging.getLogger("asyncio").setLevel(logging.ERROR)


# Update this path to your checkpoint directory
# Recommended: Create a symlink in weights/ directory and use relative path
CHECKPOINT_PATH = "../weights/gpt-oss-20b"
# Example: ln -s /path/to/your/checkpoint weights/gpt-oss-20b

# Alternative: Use absolute path directly
# CHECKPOINT_PATH = "/absolute/path/to/checkpoint"



checkpoint_path = Path(CHECKPOINT_PATH).expanduser().resolve()

# Verify path exists
if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint path does not exist: {checkpoint_path}\n"
        f"Please update CHECKPOINT_PATH to point to your checkpoint directory."
    )

def detect_checkpoint_format(checkpoint_path: Path) -> str:
    """Detect checkpoint format (Orbax or SafeTensors)."""
    # Check if path exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Check for Orbax checkpoint: should have a "0" subdirectory with state
    if (checkpoint_path / "0").exists():
        state_path = checkpoint_path / "0" / "state"
        if state_path.exists() and (state_path / "_METADATA").exists():
            return "orbax"
        # Alternative: check if state directory exists directly
        elif state_path.exists():
            # Check for any metadata files that indicate Orbax
            if any(state_path.glob("*METADATA*")) or any(state_path.glob("*.pkl")):
                return "orbax"
    
    # Check for SafeTensors: should have .safetensors files
    safetensor_files = list(checkpoint_path.glob("*.safetensors"))
    if safetensor_files:
        return "safetensors"
    
    # Provide helpful error message with diagnostics
    print(f"\n⚠️  Could not detect checkpoint format. Diagnostics:")
    print(f"   Path: {CHECKPOINT_PATH}")
    print(f"   Path exists: {checkpoint_path.exists()}")
    
    if checkpoint_path.is_dir():
        print(f"   Contents:")
        try:
            contents = list(checkpoint_path.iterdir())[:10]  # First 10 items
            for item in contents:
                print(f"     - {item.name} ({'dir' if item.is_dir() else 'file'})")
        except PermissionError:
            print(f"     (Permission denied)")
    
    # Check for common issues
    if (checkpoint_path / "0").exists():
        print(f"   Found '0' subdirectory, checking structure...")
        zero_dir = checkpoint_path / "0"
        if (zero_dir / "state").exists():
            state_contents = list((zero_dir / "state").iterdir())[:5]
            print(f"   State directory contents: {[c.name for c in state_contents]}")
    
    raise ValueError(
        f"Could not detect checkpoint format in {checkpoint_path}\n"
        f"Expected either:\n"
        f"  - Orbax: checkpoint/0/state/_METADATA exists\n"
        f"  - SafeTensors: checkpoint/*.safetensors files exist\n"
        f"Please verify your checkpoint path is correct."
    )

# Detect format
checkpoint_format = detect_checkpoint_format(checkpoint_path)
print(f"✓ Detected checkpoint format: {checkpoint_format}")
print(f"✓ Checkpoint path: {CHECKPOINT_PATH}")

# Load checkpoint
print(f"\nLoading checkpoint...")
load_start = time.time()

if checkpoint_format == "orbax":
    loader = OrbaxWeightLoader(str(checkpoint_path))
    params = loader.load_params(show_progress=True, unpack_quantized=True)
    # Load config from Orbax (hardcoded for GPT-OSS-20B)
    config_dict = load_config_from_orbax(str(checkpoint_path))

    config = ModelConfig(
        num_hidden_layers=config_dict["num_hidden_layers"],
        hidden_size=config_dict["hidden_size"],
        head_dim=config_dict["head_dim"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        sliding_window=config_dict["sliding_window"],
        intermediate_size=config_dict["intermediate_size"],
        num_experts=config_dict["num_experts"],
        experts_per_token=config_dict["experts_per_token"],
        vocab_size=config_dict["vocab_size"],
        swiglu_limit=config_dict["swiglu_limit"],
        rope_theta=config_dict["rope_theta"],
        rope_scaling_factor=config_dict["rope_scaling_factor"],
        rope_ntk_alpha=config_dict["rope_ntk_alpha"],
        rope_ntk_beta=config_dict["rope_ntk_beta"],
        initial_context_length=config_dict["initial_context_length"],
    )
else:
    # SafeTensors: load config first, then weights
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(
            num_hidden_layers=config_dict.get("num_hidden_layers", 36),
            hidden_size=config_dict.get("hidden_size", 2880),
            head_dim=config_dict.get("head_dim", 64),
            num_attention_heads=config_dict.get("num_attention_heads", 64),
            num_key_value_heads=config_dict.get("num_key_value_heads", 8),
            sliding_window=config_dict.get("sliding_window", 128),
            intermediate_size=config_dict.get("intermediate_size", 2880),
            num_experts=config_dict.get("num_experts", 128),
            experts_per_token=config_dict.get("experts_per_token", 4),
            vocab_size=config_dict.get("vocab_size", 201088),
            swiglu_limit=config_dict.get("swiglu_limit", 7.0),
            rope_theta=config_dict.get("rope_theta", 150000.0),
            rope_scaling_factor=config_dict.get("rope_scaling_factor", 32.0),
            rope_ntk_alpha=config_dict.get("rope_ntk_alpha", 1.0),
            rope_ntk_beta=config_dict.get("rope_ntk_beta", 32.0),
            initial_context_length=config_dict.get("initial_context_length", 4096),
        )
    else:
        # Fallback to defaults
        config = ModelConfig()
    
    loader = WeightLoader(str(checkpoint_path))
    params = loader.load_params(config, show_progress=True)

load_time = time.time() - load_start
print(f"\n✓ Checkpoint loaded in {load_time:.2f}s")
```

    ✓ Detected checkpoint format: orbax
    ✓ Checkpoint path: ../weights/gpt-oss-20b
    
    Loading checkpoint...


    Loading checkpoint: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100/100% [00:05]

      ✓ Loaded 27 top-level parameter groups
      ✓ Total parameters: 20,914,757,184 (20.91B)
    
    ✓ Checkpoint loaded in 6.00s


    


## Model Initialization

Initialize the Transformer model with the GPT-OSS-20B configuration:

- **24 transformer layers** (for Orbax checkpoints)
- **2880 hidden dimensions**
- **64 attention heads** (8 key/value heads with GQA)
- **32 MoE experts** (4 experts per token)
- **201,088 vocabulary size**

Also initialize the tokenizer for encoding/decoding text.


```python
# Create model with config
model = Transformer(config=config)

# Get tokenizer (uses openai-harmony if available, otherwise manual construction)
tokenizer = get_tokenizer()

print(f"✓ Model initialized")
print(f"  Layers: {config.num_hidden_layers}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Attention heads: {config.num_attention_heads} (Q), {config.num_key_value_heads} (K/V)")
print(f"  MoE experts: {config.num_experts} (activating {config.experts_per_token} per token)")
print(f"  Vocabulary size: {config.vocab_size}")
print(f"✓ Tokenizer loaded")
```

    ✓ Model initialized
      Layers: 24
      Hidden size: 2880
      Attention heads: 64 (Q), 8 (K/V)
      MoE experts: 32 (activating 4 per token)
      Vocabulary size: 201088
    ✓ Tokenizer loaded


## Simple Generation

Generate text using greedy decoding (temperature=0.0). The first token will include JIT compilation time, which is cached for subsequent runs.


```python
# Set up prompt
prompt = "Who wrote Romeo and Juliet?"
prompt_tokens = tokenizer.encode(prompt)

print(f"Prompt: {prompt}")
print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:10]}...")

# Create RNG key (not needed for greedy, but included for consistency)
rng_key = jax.random.PRNGKey(42)

# Generate tokens
print(f"\nGenerating tokens...")
gen_start = time.time()

output_tokens, stats = generate(
    model=model,
    params=params,
    prompt_tokens=prompt_tokens,
    max_new_tokens=10,  # Short output
    temperature=0.0,  # Greedy decoding
    rng_key=rng_key,
    config=config,
    use_kv_cache=True,
    show_progress=True,
    return_stats=True
)

gen_time = time.time() - gen_start

# Decode output
output_text = tokenizer.decode(output_tokens)
generated_tokens = output_tokens[len(prompt_tokens):]

print(f"\n{'='*60}")
print(f"Output:")
print(f"{output_text}")
print(f"\n{'='*60}")
print(f"Statistics:")
print(f"  Generated tokens: {len(generated_tokens)}")
print(f"  Total time: {stats['total_time']:.2f}s")
print(f"  First token time (TTFT): {stats['first_token_time']:.2f}s")
print(f"  Tokens/second: {stats['tokens_per_second']:.2f}")
if stats['tokens_per_second_after_first'] > 0:
    print(f"  Tokens/second (after first): {stats['tokens_per_second_after_first']:.2f}")
```

    Prompt: Who wrote Romeo and Juliet?
    Prompt tokens (6): [20600, 11955, 96615, 326, 128971, 30]...
    
    Generating tokens...
    [KV Cache] Initialized 24 caches, shape: (1, 4096, 8, 64)


    Generating:  10%|█████▉                                                     | 1/10 [00:06<00:55,  6.16s/tok, last_token=410, ttft=6.15s]

    
    [Token 0] Detailed timing, cache_offset=6:
      Input shape: (6,)
      Array creation: 3.52ms
      Forward pass: 6.14s
      Total token time: 6.15s


    Generating:  20%|███████████▌                                              | 2/10 [00:08<00:30,  3.82s/tok, last_token=4066, tok_s=0.24]

    
    [Token 1] Detailed timing, cache_offset=7:
      Input shape: (1,)
      Array creation: 3.31ms
      Forward pass: 2.17s
      Total token time: 2.18s


    Generating:  30%|█████████████████▋                                         | 3/10 [00:09<00:18,  2.61s/tok, last_token=256, tok_s=0.32]

    
    [Token 2] Detailed timing, cache_offset=8:
      Input shape: (1,)
      Array creation: 0.12ms
      Forward pass: 1.16s
      Total token time: 1.16s


    Generating: 100%|███████████████████████████████████████████████████████████| 10/10 [00:17<00:00,  1.80s/tok, last_token=17, tok_s=0.56]

    
    ============================================================
    Output:
    Who wrote Romeo and Juliet?**  
       *Answer: William Shakespeare*
    
    2
    
    ============================================================
    Statistics:
      Generated tokens: 10
      Total time: 17.95s
      First token time (TTFT): 6.15s
      Tokens/second: 0.56
      Tokens/second (after first): 0.76


    


## Streaming Generation with Progress Bar

Generate tokens with a progress bar and detailed statistics. The `generate()` function supports `show_progress=True` and `return_stats=True` for real-time feedback.


```python
def generate_with_progress(model, params, prompt_tokens, max_new_tokens=100, 
                           temperature=0.8, config=None):
    """Generate tokens with progress bar using existing generate() function."""
    rng_key = jax.random.PRNGKey(42)
    
    # Track timing
    start_time = time.time()
    first_token_time = None
    
    # Use token_callback to track first token
    def token_callback(token):
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time() - start_time
    
    # Generate with progress bar
    output_tokens, stats = generate(
        model=model,
        params=params,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng_key=rng_key,
        config=config,
        use_kv_cache=True,
        show_progress=True,
        token_callback=token_callback,
        return_stats=True
    )
    
    # Extract stats
    result_stats = {
        "first_token_time": first_token_time or stats.get('first_token_time', 0),
        "total_time": stats['total_time'],
        "tokens_per_second": stats['tokens_per_second'],
        "tokens_per_second_after_first": stats.get('tokens_per_second_after_first', 0),
    }
    
    return output_tokens, result_stats

# Test with a new prompt
prompt = "The future of artificial intelligence"
prompt_tokens = tokenizer.encode(prompt)

print(f"Prompt: {prompt}")
print(f"\nGenerating with temperature={0.8}...")

output_tokens, stats = generate_with_progress(
    model=model,
    params=params,
    prompt_tokens=prompt_tokens,
    max_new_tokens=50,
    temperature=0.8,
    config=config
)

# Decode and display
output_text = tokenizer.decode(output_tokens)
generated_tokens = output_tokens[len(prompt_tokens):]

print(f"\n{'='*60}")
print(f"Generated text:")
print(f"{output_text}")
print(f"\n{'='*60}")
print(f"Performance Statistics:")
print(f"  First token time (TTFT): {stats['first_token_time']:.3f}s")
print(f"  Total generation time: {stats['total_time']:.2f}s")
print(f"  Tokens/second: {stats['tokens_per_second']:.2f}")
if stats['tokens_per_second_after_first'] > 0:
    print(f"  Tokens/second (after first): {stats['tokens_per_second_after_first']:.2f}")
```

    Prompt: The future of artificial intelligence
    
    Generating with temperature=0.8...
    [KV Cache] Initialized 24 caches, shape: (1, 4096, 8, 64)


    Generating:   2%|█▏                                                         | 1/50 [00:05<04:26,  5.43s/tok, last_token=350, ttft=5.26s]

    
    [Token 0] Detailed timing, cache_offset=5:
      Input shape: (5,)
      Array creation: 5.71ms
      Forward pass: 5.25s
      Total token time: 5.26s


    Generating:   4%|██▎                                                      | 2/50 [00:06<02:17,  2.87s/tok, last_token=17527, tok_s=0.32]

    
    [Token 1] Detailed timing, cache_offset=6:
      Input shape: (1,)
      Array creation: 0.29ms
      Forward pass: 1.07s
      Total token time: 1.07s


    Generating:   6%|███▋                                                         | 3/50 [00:07<01:30,  1.93s/tok, last_token=8, tok_s=0.42]

    
    [Token 2] Detailed timing, cache_offset=7:
      Input shape: (1,)
      Array creation: 0.11ms
      Forward pass: 0.81s
      Total token time: 0.81s


    Generating: 100%|██████████████████████████████████████████████████████████| 50/50 [01:06<00:00,  1.32s/tok, last_token=410, tok_s=0.76]

    
    ============================================================
    Generated text:
    The future of artificial intelligence (AI) is a topic that has garnered significant attention in both the scientific community and popular media.
    
    In this article, we will explore the current state of AI research and development, as well as the potential applications and challenges that lie ahead.
    
    **
    
    ============================================================
    Performance Statistics:
      First token time (TTFT): 5.440s
      Total generation time: 66.14s
      Tokens/second: 0.76
      Tokens/second (after first): 0.81


    


## Temperature Sampling

Experiment with different temperature values to control generation randomness:

- **Temperature = 0.0**: Greedy decoding (deterministic, always picks most likely token)
- **Temperature = 0.5**: Low randomness (mostly deterministic with slight variation)
- **Temperature = 1.0**: Balanced randomness (default for most use cases)
- **Temperature = 1.5**: High randomness (more creative but potentially less coherent)


```python
prompt = "Once upon a time"
prompt_tokens = tokenizer.encode(prompt)

temperatures = [0.0, 0.5, 1.0, 1.5]

print(f"Prompt: '{prompt}'")
print(f"\nGenerating with different temperatures...")
print(f"{'='*60}")

for temp in temperatures:
    print(f"\nTemperature = {temp}:")
    print("-" * 60)
    
    output_tokens, stats = generate(
        model=model,
        params=params,
        prompt_tokens=prompt_tokens,
        max_new_tokens=30,
        temperature=temp,
        rng_key=jax.random.PRNGKey(42 + int(temp * 100)),  # Different seed per temp
        config=config,
        use_kv_cache=True,
        show_progress=False,  # Disable progress bar for cleaner output
        return_stats=True
    )
    
    output_text = tokenizer.decode(output_tokens)
    generated_text = output_text[len(prompt):].strip()
    
    print(f"Generated: {generated_text}")
    print(f"Time: {stats['total_time']:.2f}s, Speed: {stats['tokens_per_second']:.2f} tok/s")
```

    Prompt: 'Once upon a time'
    
    Generating with different temperatures...
    ============================================================
    
    Temperature = 0.0:
    ------------------------------------------------------------
    
    [Token 0] Detailed timing, cache_offset=4:
      Input shape: (4,)
      Array creation: 5.99ms
      Forward pass: 4.59s
      Total token time: 4.59s
    
    [Token 1] Detailed timing, cache_offset=5:
      Input shape: (1,)
      Array creation: 0.05ms
      Forward pass: 1.15s
      Total token time: 1.15s
    
    [Token 2] Detailed timing, cache_offset=6:
      Input shape: (1,)
      Array creation: 0.04ms
      Forward pass: 0.85s
      Total token time: 0.85s
    Generated: , in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him
    Time: 29.48s, Speed: 1.02 tok/s
    
    Temperature = 0.5:
    ------------------------------------------------------------
    
    [Token 0] Detailed timing, cache_offset=4:
      Input shape: (4,)
      Array creation: 0.05ms
      Forward pass: 2.89s
      Total token time: 2.89s
    
    [Token 1] Detailed timing, cache_offset=5:
      Input shape: (1,)
      Array creation: 0.06ms
      Forward pass: 0.85s
      Total token time: 0.85s
    
    [Token 2] Detailed timing, cache_offset=6:
      Input shape: (1,)
      Array creation: 0.05ms
      Forward pass: 0.85s
      Total token time: 0.85s
    Generated: , in a small town named Willowbrook, there lived a young girl named Lily. Lily had a curious mind and a heart full of kindness. She
    Time: 27.65s, Speed: 1.09 tok/s
    
    Temperature = 1.0:
    ------------------------------------------------------------
    
    [Token 0] Detailed timing, cache_offset=4:
      Input shape: (4,)
      Array creation: 0.04ms
      Forward pass: 2.88s
      Total token time: 2.88s
    
    [Token 1] Detailed timing, cache_offset=5:
      Input shape: (1,)
      Array creation: 0.06ms
      Forward pass: 0.85s
      Total token time: 0.85s
    
    [Token 2] Detailed timing, cache_offset=6:
      Input shape: (1,)
      Array creation: 0.04ms
      Forward pass: 0.85s
      Total token time: 0.85s
    Generated: , a group of friends decided to hold a competition to see who could come up with the most creative way to say "I love you." They each
    Time: 27.72s, Speed: 1.08 tok/s
    
    Temperature = 1.5:
    ------------------------------------------------------------
    
    [Token 0] Detailed timing, cache_offset=4:
      Input shape: (4,)
      Array creation: 0.05ms
      Forward pass: 2.90s
      Total token time: 2.90s
    
    [Token 1] Detailed timing, cache_offset=5:
      Input shape: (1,)
      Array creation: 0.06ms
      Forward pass: 0.86s
      Total token time: 0.86s
    
    [Token 2] Detailed timing, cache_offset=6:
      Input shape: (1,)
      Array creation: 0.04ms
      Forward pass: 0.85s
      Total token time: 0.85s
    Generated: , a developer discovered a code that could automatically send an email and convert each instance into a PDF. To unlock the secret, he decided to go with
    Time: 27.78s, Speed: 1.08 tok/s


## OpenAI Harmony Prompt Format Example

A simple example demonstrating proper Harmony tokenizer formatting with special tokens (`<|startoftext|>`, `<|message|>`, etc.):


```python
# Harmony-formatted prompt example with colored output
# Import necessary modules
import jax
import re
from IPython.display import HTML, display
from gpt_oss.jax.inference import generate
from gpt_oss.tokenizer import get_tokenizer

# Initialize tokenizer
tokenizer = get_tokenizer()

# User message
user_message = "What is the capital of France?"

print("=" * 70)
print("Harmony Format Example with Color-Coded Output")
print("=" * 70)

# Display a) Original user message
display(HTML(f"""
<div style="margin: 15px 0; padding: 15px; background-color: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px;">
    <h3 style="margin-top: 0; color: #1976d2; font-family: Arial, sans-serif;">
        a) Original User Message
    </h3>
    <p style="font-size: 16px; font-family: monospace; color: #0d47a1; margin: 0;">
        {user_message}
    </p>
</div>
"""))

# Try to use openai-harmony for proper Harmony formatting
try:
    from openai_harmony import (
        load_harmony_encoding,
        HarmonyEncodingName,
        Conversation,
        Message,
        Role
    )
    
    # Load Harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    # Create Harmony conversation
    conversation = Conversation.from_messages([
        Message.from_role_and_content(Role.USER, user_message)
    ])
    
    # Render conversation for completion (adds Harmony special tokens)
    prompt_tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    
    # Decode prompt to show Harmony formatting
    prompt_text = tokenizer.decode(prompt_tokens)
    
    # Display b) Harmony-formatted prompt
    # Escape HTML special characters but preserve Harmony tokens
    prompt_text_escaped = prompt_text.replace('<', '&lt;').replace('>', '&gt;')
    
    display(HTML(f"""
<div style="margin: 15px 0; padding: 15px; background-color: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 4px;">
    <h3 style="margin-top: 0; color: #2e7d32; font-family: Arial, sans-serif;">
        b) Harmony-Formatted Prompt (with special tokens)
    </h3>
    <p style="font-size: 14px; font-family: monospace; color: #1b5e20; margin: 0; white-space: pre-wrap; word-break: break-all;">
        {prompt_text_escaped}
    </p>
    <p style="font-size: 12px; color: #558b2f; margin-top: 10px; font-family: Arial, sans-serif;">
        ✓ Prompt tokens: {len(prompt_tokens)} | Special tokens: {[t for t in prompt_tokens if t >= 199998]}
    </p>
</div>
"""))
    
    # Get Harmony stop tokens
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()
    
    # Ensure model, params, config are available
    if "model" in globals() and "params" in globals() and "config" in globals():
        print("\nGenerating response...")
        
        output_tokens, stats = generate(
            model=model,
            params=params,
            prompt_tokens=prompt_tokens,
            max_new_tokens=50,
            temperature=0.0,  # Greedy decoding
            rng_key=jax.random.PRNGKey(42),
            config=config,
            use_kv_cache=True,
            show_progress=False,
            return_stats=True
        )
        
        # Filter out stop tokens
        filtered_tokens = []
        for token in output_tokens[len(prompt_tokens):]:
            if token in stop_token_ids:
                break
            filtered_tokens.append(token)
        
        # Extract just the generated part (without the prompt)
        generated_only = tokenizer.decode(filtered_tokens)
        generated_only_escaped = generated_only.replace('<', '&lt;').replace('>', '&gt;')
        
        # Display c) Full generated response
        display(HTML(f"""
<div style="margin: 15px 0; padding: 15px; background-color: #fff9c4; border-left: 4px solid #fbc02d; border-radius: 4px;">
    <h3 style="margin-top: 0; color: #f57f17; font-family: Arial, sans-serif;">
        c) Full Generated Response (raw Harmony format)
    </h3>
    <p style="font-size: 14px; font-family: monospace; color: #827717; margin: 0; white-space: pre-wrap; word-break: break-all;">
        {generated_only_escaped}
    </p>
</div>
"""))
        
        # Parse Harmony output to separate reasoning from final answer
        try:
            # Parse completion tokens into structured messages
            completion_tokens = filtered_tokens
            messages = encoding.parse_messages_from_completion_tokens(completion_tokens, Role.ASSISTANT)
            
            # Extract reasoning and final answer from parsed messages
            reasoning_parts = []
            answer_parts = []
            
            for msg in messages:
                # Check message structure - Harmony messages have recipient/channel info
                msg_dict = msg.to_dict() if hasattr(msg, 'to_dict') else {}
                
                # Look for channel information
                recipient = getattr(msg, 'recipient', None) or msg_dict.get('recipient', '')
                content = getattr(msg, 'content', None) or msg_dict.get('content', '')
                
                if 'analysis' in str(recipient).lower() or 'analysis' in str(content).lower():
                    reasoning_parts.append(str(content))
                elif 'main' in str(recipient).lower() or 'final' in str(recipient).lower() or (not recipient and content):
                    answer_parts.append(str(content))
                else:
                    # Try to extract from content string
                    content_str = str(content)
                    if '<|channel|>analysis' in content_str:
                        reasoning_parts.append(content_str)
                    elif '<|channel|>main' in content_str or '<|channel|>final' in content_str:
                        answer_parts.append(content_str)
            
        except Exception as e:
            # If parsing fails, try manual parsing
            print(f"Harmony parsing failed ({e}), using manual parsing...")
            
            # Extract reasoning (analysis channel)
            reasoning_match = re.search(r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)', generated_only, re.DOTALL)
            if reasoning_match:
                reasoning_parts = [reasoning_match.group(1).strip()]
            
            # Extract final answer (main/final channel)
            answer_match = re.search(r'<\|channel\|>(main|final)<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)', generated_only, re.DOTALL)
            if answer_match:
                answer_parts = [answer_match.group(2).strip()]
        
        # Display d) Parsed output
        display(HTML(f"""
<div style="margin: 15px 0;">
    <h3 style="color: #424242; font-family: Arial, sans-serif; margin-bottom: 10px;">
        d) Parsed Harmony Output (extracted channels)
    </h3>
</div>
"""))
        
        # Display reasoning (green)
        if reasoning_parts:
            reasoning_html = "<br>".join([f"<p style='margin: 5px 0;'>{str(part)}</p>" for part in reasoning_parts])
            display(HTML(f"""
<div style="margin: 15px 0; padding: 15px; background-color: #c8e6c9; border-left: 4px solid #66bb6a; border-radius: 4px;">
    <h4 style="margin-top: 0; color: #2e7d32; font-family: Arial, sans-serif;">
        📊 Reasoning/Analysis (analysis channel)
    </h4>
    <div style="font-size: 15px; color: #1b5e20; font-family: Arial, sans-serif;">
        {reasoning_html}
    </div>
</div>
"""))
        
        # Display final answer (magenta/purple)
        if answer_parts:
            answer_html = "<br>".join([f"<p style='margin: 5px 0;'>{str(part)}</p>" for part in answer_parts])
            display(HTML(f"""
<div style="margin: 15px 0; padding: 15px; background-color: #f3e5f5; border-left: 4px solid #ab47bc; border-radius: 4px;">
    <h4 style="margin-top: 0; color: #7b1fa2; font-family: Arial, sans-serif;">
        💬 Final Answer (final channel)
    </h4>
    <div style="font-size: 15px; color: #4a148c; font-family: Arial, sans-serif;">
        {answer_html}
    </div>
</div>
"""))
        
        # Display stats
        print(f"\nPerformance: {stats['total_time']:.2f}s | {stats['tokens_per_second']:.2f} tok/s")
    
    else:
        print("\n⚠️  Please run the Model Initialization and Checkpoint Loading sections first.")
        
except ImportError:
    print("⚠️  openai-harmony not available")
    print("   Install with: pip install openai-harmony")
```

    ======================================================================
    Harmony Format Example with Color-Coded Output
    ======================================================================




<div style="margin: 15px 0; padding: 15px; background-color: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px;">
    <h3 style="margin-top: 0; color: #1976d2; font-family: Arial, sans-serif;">
        a) Original User Message
    </h3>
    <p style="font-size: 16px; font-family: monospace; color: #0d47a1; margin: 0;">
        What is the capital of France?
    </p>
</div>





<div style="margin: 15px 0; padding: 15px; background-color: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 4px;">
    <h3 style="margin-top: 0; color: #2e7d32; font-family: Arial, sans-serif;">
        b) Harmony-Formatted Prompt (with special tokens)
    </h3>
    <p style="font-size: 14px; font-family: monospace; color: #1b5e20; margin: 0; white-space: pre-wrap; word-break: break-all;">
        &lt;|start|&gt;user&lt;|message|&gt;What is the capital of France?&lt;|end|&gt;&lt;|start|&gt;assistant
    </p>
    <p style="font-size: 12px; color: #558b2f; margin-top: 10px; font-family: Arial, sans-serif;">
        ✓ Prompt tokens: 13 | Special tokens: [200006, 200008, 200007, 200006]
    </p>
</div>



    
    Generating response...
    
    [Token 0] Detailed timing, cache_offset=13:
      Input shape: (13,)
      Array creation: 6.64ms
      Forward pass: 14.84s
      Total token time: 14.85s
    
    [Token 1] Detailed timing, cache_offset=14:
      Input shape: (1,)
      Array creation: 0.05ms
      Forward pass: 0.85s
      Total token time: 0.86s
    
    [Token 2] Detailed timing, cache_offset=15:
      Input shape: (1,)
      Array creation: 0.04ms
      Forward pass: 0.85s
      Total token time: 0.85s




<div style="margin: 15px 0; padding: 15px; background-color: #fff9c4; border-left: 4px solid #fbc02d; border-radius: 4px;">
    <h3 style="margin-top: 0; color: #f57f17; font-family: Arial, sans-serif;">
        c) Full Generated Response (raw Harmony format)
    </h3>
    <p style="font-size: 14px; font-family: monospace; color: #827717; margin: 0; white-space: pre-wrap; word-break: break-all;">
        &lt;|channel|&gt;analysis&lt;|message|&gt;We need to answer the question: "What is the capital of France?" The answer: Paris.&lt;|end|&gt;&lt;|start|&gt;assistant&lt;|channel|&gt;final&lt;|message|&gt;The capital of France is **Paris**.
    </p>
</div>





<div style="margin: 15px 0;">
    <h3 style="color: #424242; font-family: Arial, sans-serif; margin-bottom: 10px;">
        d) Parsed Harmony Output (extracted channels)
    </h3>
</div>





<div style="margin: 15px 0; padding: 15px; background-color: #f3e5f5; border-left: 4px solid #ab47bc; border-radius: 4px;">
    <h4 style="margin-top: 0; color: #7b1fa2; font-family: Arial, sans-serif;">
        💬 Final Answer (final channel)
    </h4>
    <div style="font-size: 15px; color: #4a148c; font-family: Arial, sans-serif;">
        <p style='margin: 5px 0;'>[TextContent(text='We need to answer the question: "What is the capital of France?" The answer: Paris.')]</p><br><p style='margin: 5px 0;'>[TextContent(text='The capital of France is **Paris**.')]</p>
    </div>
</div>



    
    Performance: 61.38s | 0.81 tok/s


## Conclusion

This notebook demonstrated:

1. **Checkpoint loading**: Fast loading from both Orbax and SafeTensors formats
2. **Model initialization**: Setting up the GPT-OSS-20B transformer model
3. **Text generation**: Greedy and temperature-based sampling
4. **Performance tracking**: Measuring TTFT (time to first token) and tokens/second
5. **Compilation caching**: Leveraging JAX's compilation cache for faster subsequent runs

### Next Steps

- **Milestone 3**: Chat integration with streaming responses
- **Milestone 4**: PyPI package preparation
- **Milestone 5**: Advanced optimizations (FlashAttention, quantization)

### Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [GPT-OSS-20B Model Card](https://huggingface.co/atsentia/gpt-oss-20b)

Happy generating! 🚀
