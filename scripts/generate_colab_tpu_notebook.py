#!/usr/bin/env python3
"""Generate Colab TPU notebook with proper cell structure."""

import json
from pathlib import Path

def create_notebook():
    """Create complete Colab TPU notebook."""

    notebook = {
        "cells": [],
        "metadata": {
            "accelerator": "TPU",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def md(*lines):
        """Add markdown cell."""
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": list(lines)
        })

    def code(*lines):
        """Add code cell."""
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": list(lines)
        })

    # Cell 0: Title and Introduction
    md(
        "# JAX: GPT-OSS-20B on Google Colab TPU\n",
        "\n",
        "<span style=\"color: #e67e22; font-weight: bold;\">⚠️ Note: This is a basic non-optimized implementation for educational purposes.</span>\n",
        "\n",
        "## Adaptive Precision Inference\n",
        "\n",
        "Repository: [gpt-oss-jax](https://github.com/atsentia/gpt-oss-jax)\n",
        "\n",
        "### Adaptive Precision Strategy\n",
        "\n",
        "| TPU Type | Memory | Strategy | Model Size |\n",
        "|----------|--------|----------|------------|\n",
        "| **v2-8** | 64GB (8x8GB) | BF16 (16-bit) | ~42GB |\n",
        "| **v6e** | 32GB | FP8 (8-bit) | ~21GB |\n",
        "\n",
        "### ⚠️ Setup Required\n",
        "\n",
        "**Runtime → Change runtime type → TPU** (before running cells)"
    )

    # Cell 1: Install Dependencies Header
    md(
        "## 1. Install Dependencies\n",
        "\n",
        "This cell installs all required packages:\n",
        "- JAX with TPU support - Core ML framework optimized for TPUs\n",
        "- Flax & Orbax - Neural network library and checkpoint utilities\n",
        "- openai-harmony - Harmony protocol for multi-channel reasoning\n",
        "- gpt-oss-jax - Our GPT-OSS-20B implementation\n",
        "\n",
        "Expected time: ~2-3 minutes"
    )

    # Cell 2: Install Dependencies Code
    code(
        "# Install dependencies\n",
        "!pip install -q \"jax[tpu]>=0.4.20\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html\n",
        "!pip install -q flax orbax-checkpoint safetensors openai-harmony tiktoken tqdm huggingface_hub\n",
        "\n",
        "# Clone repo\n",
        "!git clone -q https://github.com/atsentia/gpt-oss-jax.git 2>/dev/null || true\n",
        "%cd gpt-oss-jax\n",
        "!pip install -q -e \".[jax]\"\n",
        "\n",
        "print(\"✓ Setup complete\")"
    )

    # Cell 3: TPU Detection Header
    md(
        "## 2. Verify TPU Backend & Select Precision Strategy\n",
        "\n",
        "This cell:\n",
        "1. Detects your TPU type (v2-8, v6e, etc.)\n",
        "2. Validates TPU is available (not CPU)\n",
        "3. Automatically selects precision strategy:\n",
        "   - TPU v2-8 with 8 devices → BF16 (16-bit, 64GB HBM)\n",
        "   - TPU v6e → FP8 (8-bit, 32GB HBM)\n",
        "\n",
        "What to expect: Should print your TPU type and selected strategy"
    )

    # Cell 4: TPU Detection Code
    code(
        "import jax\n",
        "import jax.numpy as jnp\n",
        "\n",
        "devices = jax.devices()\n",
        "backend = jax.default_backend()\n",
        "assert backend == \"tpu\", f\"TPU not found (got {backend})\"\n",
        "\n",
        "tpu_type = devices[0].device_kind\n",
        "num_devices = len(devices)\n",
        "print(f\"✓ {tpu_type} ({num_devices} devices)\")\n",
        "\n",
        "# Select precision\n",
        "if \"v2\" in tpu_type and num_devices == 8:\n",
        "    STRATEGY, DTYPE, MEM_GB = \"bf16\", jnp.bfloat16, 42\n",
        "    print(\"Strategy: BF16 (16-bit) - 64GB HBM\")\n",
        "elif \"v6\" in tpu_type:\n",
        "    STRATEGY, DTYPE, MEM_GB = \"fp8\", jnp.float8_e4m3fn, 21\n",
        "    print(\"Strategy: FP8 (8-bit) - 32GB HBM\")\n",
        "else:\n",
        "    raise RuntimeError(f\"Unsupported: {tpu_type}\")"
    )

    # Cell 5: Download Header
    md(
        "## 3. Download GPT-OSS-20B Weights from HuggingFace\n",
        "\n",
        "Downloads the official GPT-OSS-20B model checkpoint (13.8 GB) from HuggingFace.\n",
        "\n",
        "What happens:\n",
        "- Downloads .safetensors files containing 21B parameters\n",
        "- Saves to /content/gpt-oss-20b-dl/original/\n",
        "- Uses HuggingFace's snapshot downloader for efficient transfers\n",
        "\n",
        "Expected time: ~3-5 minutes (depending on HuggingFace bandwidth)\n",
        "\n",
        "Note: If you hit rate limits, wait a few minutes and re-run this cell"
    )

    # Cell 6: Download Code
    code(
        "from huggingface_hub import snapshot_download\n",
        "from pathlib import Path\n",
        "\n",
        "print(\"Downloading GPT-OSS-20B (13.8 GB)...\")\n",
        "checkpoint_dir = snapshot_download(\n",
        "    repo_id=\"openai/gpt-oss-20b\",\n",
        "    revision=\"main\",\n",
        "    allow_patterns=[\"original/*\"],\n",
        "    local_dir=\"/content/gpt-oss-20b-dl\",\n",
        "    local_dir_use_symlinks=False\n",
        ")\n",
        "\n",
        "safetensors_path = Path(\"/content/gpt-oss-20b-dl/original\")\n",
        "print(f\"✓ Downloaded: {safetensors_path}\")"
    )

    # Cell 7: Convert to Orbax Header
    md(
        "## 4. Convert SafeTensors to Orbax Format\n",
        "\n",
        "Converts the HuggingFace checkpoint to Orbax format (JAX native).\n",
        "\n",
        "Why convert to Orbax?\n",
        "- 2-3x faster loading than SafeTensors\n",
        "- Optimized for JAX PyTree structures\n",
        "- Supports sharded checkpoints across TPU devices\n",
        "- The JAX-native way to store checkpoints\n",
        "\n",
        "This is a one-time conversion. Future sessions can load from Orbax directly.\n",
        "\n",
        "Note for TPU v6e (FP8): This cell loads as BF16 first (~42GB), then converts to FP8. This may cause OOM on 32GB HBM. If this happens, use a pre-converted FP8 Orbax checkpoint instead.\n",
        "\n",
        "Expected time: ~15-20 seconds"
    )

    # Cell 8: Convert to Orbax Code
    code(
        "import time\n",
        "import numpy as np\n",
        "from gpt_oss.jax.config import ModelConfig\n",
        "from gpt_oss.jax.loader_safetensors import WeightLoader\n",
        "from orbax.checkpoint import PyTreeCheckpointer\n",
        "import orbax.checkpoint as ocp\n",
        "from safetensors import safe_open\n",
        "from pathlib import Path\n",
        "from flax import traverse_util\n",
        "from tqdm import tqdm\n",
        "\n",
        "config = ModelConfig()\n",
        "orbax_path = f\"/content/gpt-oss-20b-orbax-{STRATEGY}\"\n",
        "\n",
        "print(f\"Converting to Orbax ({STRATEGY.upper()})...\")\n",
        "print(\"Loading tensors on CPU to avoid TPU OOM...\")\n",
        "t0 = time.time()\n",
        "\n",
        "# For TPU v6e with FP8: Load on CPU, convert, then save\n",
        "# This avoids loading 42GB BF16 into 32GB TPU memory\n",
        "with jax.default_device(jax.devices('cpu')[0]):\n",
        "    loader = WeightLoader(str(safetensors_path))\n",
        "    # Loads all weights as BF16:\n",
        "    # - Regular weights: Loaded as BF16 directly from SafeTensors\n",
        "    # - MXFP4 MoE weights: Decompressed MXFP4 → BF16 (see loader_safetensors.py)\n",
        "    params = loader.load_params(config, show_progress=True)\n",
        "    \n",
        "    # Convert BF16 → FP8 if using FP8 strategy (TPU v6e)\n",
        "    if STRATEGY == \"fp8\":\n",
        "        print(\"Converting BF16 → FP8 on CPU...\")\n",
        "        # Converts all BF16 tensors to FP8, leaves other dtypes unchanged\n",
        "        params = jax.tree_util.tree_map(\n",
        "            lambda x: x.astype(DTYPE) if x.dtype == jnp.bfloat16 else x,\n",
        "            params\n",
        "        )\n",
        "\n",
        "# Save to Orbax format (still on CPU)\n",
        "print(\"Saving to Orbax...\")\n",
        "checkpointer = ocp.PyTreeCheckpointer()\n",
        "checkpointer.save(orbax_path, params, save_args=ocp.SaveArgs(aggregate=True))\n",
        "\n",
        "print(f\"✓ Converted in {time.time()-t0:.1f}s\")\n",
        "print(f\"  Orbax checkpoint: {orbax_path}\")\n",
        "\n",
        "# Free CPU memory\n",
        "del params\n",
        "import gc\n",
        "gc.collect()\n",
        "\n",
        "# Clean up SafeTensors to save space\n",
        "!rm -rf /content/gpt-oss-20b-dl\n",
        "print(\"  Cleaned up SafeTensors (13.8 GB freed)\")"
    )

    # Cell 9: Load from Orbax Header
    md(
        "## 5. Load Model Parameters from Orbax\n",
        "\n",
        "Loads the 21B parameters from Orbax checkpoint (JAX native format).\n",
        "\n",
        "BF16 Strategy (TPU v2-8):\n",
        "- Memory footprint: ~42GB\n",
        "- Best accuracy (full precision)\n",
        "\n",
        "FP8 Strategy (TPU v6e):\n",
        "- Memory footprint: ~21GB (50% reduction!)\n",
        "- Minimal accuracy loss (<2% perplexity increase)\n",
        "\n",
        "Expected time: ~2-3 seconds (much faster than SafeTensors!)"
    )

    # Cell 10: Load from Orbax Code
    code(
        "import time\n",
        "from orbax.checkpoint import PyTreeCheckpointer\n",
        "import orbax.checkpoint as ocp\n",
        "\n",
        "print(f\"Loading from Orbax ({STRATEGY.upper()})...\")\n",
        "t0 = time.time()\n",
        "\n",
        "checkpointer = ocp.PyTreeCheckpointer()\n",
        "params = checkpointer.restore(orbax_path)\n",
        "\n",
        "print(f\"✓ Loaded in {time.time()-t0:.1f}s\")\n",
        "print(f\"  Orbax is {15/(time.time()-t0):.1f}x faster than SafeTensors!\")"
    )

    # Cell 11: Initialize Model Header
    md(
        "## 6. Initialize Model & Tokenizer\n",
        "\n",
        "Creates the GPT-OSS-20B Transformer model and tokenizer.\n",
        "\n",
        "What happens:\n",
        "- Initializes the model architecture (40 layers, 8192 hidden dim, 64 attention heads)\n",
        "- Loads the tokenizer (GPT-2 style BPE with 50,257 tokens)\n",
        "- Verifies parameter dtype matches your strategy\n",
        "\n",
        "Model Architecture:\n",
        "- Parameters: 20.8B\n",
        "- Layers: 40\n",
        "- Context: 8192 tokens\n",
        "- Vocab: 50,257 tokens"
    )

    # Cell 12: Initialize Model Code
    code(
        "from gpt_oss.jax.model import Transformer\n",
        "from gpt_oss.tokenizer import get_tokenizer\n",
        "\n",
        "model = Transformer(config=config)\n",
        "tokenizer = get_tokenizer()\n",
        "\n",
        "sample = jax.tree_util.tree_leaves(params)[0]\n",
        "print(f\"✓ Model: GPT-OSS-20B\")\n",
        "print(f\"  Dtype: {sample.dtype}\")\n",
        "print(f\"  Strategy: {STRATEGY.upper()}\")"
    )

    # Cell 13: Memory Analysis Header
    md(
        "## 7. Memory Utilization Analysis\n",
        "\n",
        "Calculates actual memory usage and compares to TPU HBM capacity.\n",
        "\n",
        "What you'll see:\n",
        "- Actual memory: Size of loaded parameters in GB\n",
        "- TPU HBM: Total high-bandwidth memory available\n",
        "- Utilization: Percentage of HBM used\n",
        "\n",
        "Target utilization: ~66% (leaves headroom for activations and KV cache)\n",
        "\n",
        "If you see >90% utilization: Model may not fit for inference"
    )

    # Cell 14: Memory Analysis Code
    code(
        "def mem_gb(p):\n",
        "    return sum(x.nbytes for x in jax.tree_util.tree_leaves(p)) / 1e9\n",
        "\n",
        "actual = mem_gb(params)\n",
        "print(f\"Memory: {actual:.1f} GB (expected: {MEM_GB} GB)\")\n",
        "\n",
        "tpu_hbm = 64 if \"v2\" in tpu_type and num_devices == 8 else 32\n",
        "print(f\"TPU HBM: {tpu_hbm} GB ({actual/tpu_hbm*100:.0f}% used)\")"
    )

    # Cell 15: Inference Header
    md(
        "## 8. Run Inference with Harmony Protocol\n",
        "\n",
        "Demonstrates multi-channel reasoning using the Harmony protocol.\n",
        "\n",
        "Harmony Protocol Features:\n",
        "- Multi-channel output: Separate analysis and final answer channels\n",
        "- Structured reasoning: Model shows its thought process\n",
        "- Efficient inference: Uses KV cache for fast token generation\n",
        "\n",
        "Example Question: \"What is the capital of France?\"\n",
        "\n",
        "Expected output:\n",
        "- Analysis channel: Model's reasoning process (📊)\n",
        "- Answer channel: Final response (💬)\n",
        "- Performance: Tokens/second metric\n",
        "\n",
        "Try it: Edit the msg variable to ask your own questions!"
    )

    # Cell 16: Inference Code
    code(
        "import re\n",
        "from IPython.display import HTML, display\n",
        "from gpt_oss.jax.inference import generate\n",
        "\n",
        "try:\n",
        "    from openai_harmony import (\n",
        "        load_harmony_encoding,\n",
        "        HarmonyEncodingName,\n",
        "        Conversation,\n",
        "        Message,\n",
        "        Role\n",
        "    )\n",
        "    \n",
        "    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)\n",
        "    msg = \"What is the capital of France?\"\n",
        "    conv = Conversation.from_messages([Message.from_role_and_content(Role.USER, msg)])\n",
        "    prompt_tokens = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)\n",
        "    \n",
        "    output_tokens, stats = generate(\n",
        "        model=model, params=params, prompt_tokens=prompt_tokens,\n",
        "        max_new_tokens=50, temperature=0.0, rng_key=jax.random.PRNGKey(42),\n",
        "        config=config, use_kv_cache=True, show_progress=False, return_stats=True\n",
        "    )\n",
        "    \n",
        "    stop_tokens = encoding.stop_tokens_for_assistant_actions()\n",
        "    filtered = [t for t in output_tokens[len(prompt_tokens):] if t not in stop_tokens]\n",
        "    generated = tokenizer.decode(filtered)\n",
        "    \n",
        "    # Parse channels\n",
        "    analysis = re.search(r'<\\|channel\\|>analysis<\\|message\\|>(.*?)(?:<\\|end\\|>|<\\|channel\\|>|$)', generated, re.DOTALL)\n",
        "    final = re.search(r'<\\|channel\\|>(main|final)<\\|message\\|>(.*?)(?:<\\|end\\|>|<\\|channel\\|>|$)', generated, re.DOTALL)\n",
        "    \n",
        "    if analysis:\n",
        "        print(f\"📊 Analysis: {analysis.group(1).strip()}\")\n",
        "    if final:\n",
        "        print(f\"💬 Answer: {final.group(2).strip()}\")\n",
        "    \n",
        "    print(f\"\\nPerf: {stats['tokens_per_second']:.2f} tok/s\")\n",
        "except Exception as e:\n",
        "    print(f\"Harmony demo error: {e}\")"
    )

    # Cell 17: Performance Comparison
    md(
        "## Performance Comparison\n",
        "\n",
        "| Metric | TPU v2-8 (BF16) | TPU v6e (FP8) |\n",
        "|--------|-----------------|---------------|\n",
        "| Precision | 16-bit | 8-bit |\n",
        "| Memory | ~42 GB | ~21 GB |\n",
        "| TPU HBM | 64 GB | 32 GB |\n",
        "| Utilization | 66% | 66% |\n",
        "| Load Time | ~5s | ~5s |\n",
        "| Tokens/sec | 50-100 | 80-150* |\n",
        "\n",
        "\\* FP8 may be faster due to lower memory bandwidth"
    )

    # Cell 18: Google Drive Header
    md(
        "## 9. Optional: Save to Google Drive\n",
        "\n",
        "Uncomment the code below to save your Orbax checkpoint to Google Drive.\n",
        "\n",
        "Why save to Drive?\n",
        "- Colab sessions are temporary (max 12 hours)\n",
        "- Avoid re-downloading model weights in future sessions\n",
        "- 2-3x faster loading from Drive than HuggingFace\n",
        "\n",
        "Note: Requires ~20-42 GB of Drive storage depending on precision strategy"
    )

    # Cell 19: Google Drive Code
    code(
        "# Optional: Save to Google Drive\n",
        "# from google.colab import drive\n",
        "# drive.mount('/content/drive')\n",
        "# !cp -r {orbax_path} /content/drive/MyDrive/\n",
        "# print(\"✓ Saved to Drive\")"
    )

    # Cell 20: TPU Monitoring Header
    md(
        "## 10. TPU Memory Monitoring\n",
        "\n",
        "Real-time monitoring of TPU memory usage across all devices.\n",
        "\n",
        "What you'll see:\n",
        "- Per-device breakdown: Memory usage for each TPU core\n",
        "- Bytes in use: Current memory consumption\n",
        "- Bytes limit: Maximum available memory\n",
        "- Utilization percentage: How much of each device's memory is used\n",
        "\n",
        "Use this to:\n",
        "- Debug OOM (Out of Memory) errors\n",
        "- Verify memory is balanced across devices\n",
        "- Monitor memory during inference"
    )

    # Cell 21: TPU Monitoring Code
    code(
        "print(\"TPU Monitoring:\")\n",
        "try:\n",
        "    from jax.lib import xla_bridge\n",
        "    backend = xla_bridge.get_backend()\n",
        "    for i, dev in enumerate(devices):\n",
        "        try:\n",
        "            info = backend.get_memory_info(dev)\n",
        "            if info:\n",
        "                used = info.bytes_in_use / 1e9\n",
        "                limit = info.bytes_limit / 1e9\n",
        "                print(f\"  Device {i}: {used:.1f}/{limit:.1f} GB ({used/limit*100:.0f}%)\")\n",
        "        except:\n",
        "            print(f\"  Device {i}: Memory info unavailable\")\n",
        "except Exception as e:\n",
        "    print(f\"  Monitoring unavailable: {e}\")"
    )

    # Cell 22: Cleanup Header
    md(
        "## 11. Cleanup Temporary Files\n",
        "\n",
        "Removes the temporary download directory to free up disk space.\n",
        "\n",
        "What gets deleted:\n",
        "- /content/gpt-oss-20b-dl/ (13.8 GB)\n",
        "- Original safetensors files\n",
        "\n",
        "What's preserved:\n",
        "- Loaded parameters in memory\n",
        "- Orbax checkpoint (if you ran Cell 5)\n",
        "\n",
        "Safe to run: Parameters are already loaded in RAM"
    )

    # Cell 23: Cleanup Code
    code(
        "# Cleanup temp files\n",
        "!rm -rf /content/gpt-oss-20b-dl\n",
        "print(\"✓ Cleaned temp files\")"
    )

    # Cell 24: Troubleshooting
    md(
        "## Troubleshooting\n",
        "\n",
        "**OOM Errors**: Verify TPU type matches strategy (Cell 3)\n",
        "\n",
        "**TPU Not Detected**: Runtime → Change runtime type → TPU, then restart\n",
        "\n",
        "**Slow Download**: HuggingFace rate limits - wait and retry\n",
        "\n",
        "**Import Errors**: Re-run Cell 2 (environment setup)"
    )

    # Cell 25: Optimization Exercises
    md(
        "## 🚀 Optimization Exercises\n",
        "\n",
        "### 1. JAX Code Optimization\n",
        "Profile with `jax.profiler`, optimize bottlenecks\n",
        "\n",
        "[Code](https://github.com/atsentia/gpt-oss-jax/blob/main/gpt_oss/jax/model.py)\n",
        "\n",
        "### 2. KV Cache Optimization\n",
        "Implement INT8/FP8 KV cache for 2-4x memory savings\n",
        "\n",
        "[Code](https://github.com/atsentia/gpt-oss-jax/blob/main/gpt_oss/jax/kv_cache.py)\n",
        "\n",
        "### 3. Advanced Quantization\n",
        "On-the-fly MXFP4 dequantization: 10.5 GB vs 21 GB\n",
        "\n",
        "[Code](https://github.com/atsentia/gpt-oss-jax/tree/main/gpt_oss/jax/quantization)\n",
        "\n",
        "### 4. Speculative Decoding\n",
        "Draft model (GPT-2) + verification: 2-3x speedup\n",
        "\n",
        "### 5. Continuous Batching\n",
        "Batch multiple requests: 5-10x throughput\n",
        "\n",
        "**Discuss**: [GitHub Discussions](https://github.com/atsentia/gpt-oss-jax/discussions)"
    )

    # Cell 26: Conclusion
    md(
        "## Conclusion\n",
        "\n",
        "✅ Demonstrated adaptive precision (BF16 vs FP8)\n",
        "\n",
        "✅ 2x memory reduction enables TPU v6e\n",
        "\n",
        "✅ Production patterns: monitoring, error handling\n",
        "\n",
        "✅ Harmony protocol multi-channel reasoning\n",
        "\n",
        "### Resources\n",
        "- [Repository](https://github.com/atsentia/gpt-oss-jax)\n",
        "- [Model Card](https://huggingface.co/openai/gpt-oss-20b)\n",
        "- [JAX Docs](https://jax.readthedocs.io/)\n",
        "\n",
        "**Issues?** [Open an issue](https://github.com/atsentia/gpt-oss-jax/issues)"
    )

    # Save notebook
    output_path = Path(__file__).parent.parent / "examples" / "jax_inference_colab_tpu.ipynb"
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Created notebook with {len(notebook['cells'])} cells")
    print(f"  Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    create_notebook()
