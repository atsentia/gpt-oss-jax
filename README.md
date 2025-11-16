# 🚀 JAX for GPT-OSS

JAX implementation of GPT-OSS-20B with Harmony protocol support for multi-channel reasoning.

## What is this?

This repository provides a JAX-based inference implementation for the GPT-OSS-20B language model (21B parameters). It demonstrates the **Harmony protocol**, which enables models to output structured multi-channel reasoning (separate analysis and final answer channels).

**Key Features:**
- ✅ Basic JAX inference for GPT-OSS-20B
- ✅ Harmony protocol multi-channel reasoning
- ✅ CPU and TPU backend support
- ✅ Interactive Jupyter notebooks

**Note:** This is an educational/reference implementation. For production use cases, consider optimized serving frameworks.

## Why JAX + GPT-OSS?

### Why GPT-OSS?
OpenAI's GPT-OSS models (20B and 120B parameters) are **high-quality open-weight LLMs**:
- 🏆 **Strong Performance**: Capable general-purpose language models suitable for research and applications
- 🔓 **Truly Open**: Full model weights, training code, and evaluation harnesses released
- 🎯 **Harmony Protocol**: Native support for multi-channel structured reasoning

### Why JAX?
JAX is the framework of choice for leading AI labs and production systems:
- 🏢 **Industry Adoption**: Powers Google Gemini, X (Grok), Anthropic (Claude training), Cohere
- 🔬 **Research Standard**: Preferred by DeepMind, Google Research, OpenAI (research), Allen AI
- 📊 **Ecosystem**: 1000+ JAX models on HuggingFace, extensive scientific computing libraries
- ⚡ **Performance**: XLA compilation, automatic differentiation, TPU/GPU acceleration out-of-the-box
- 🧮 **Functional Design**: Clean, composable code that scales from research prototypes to production

**Bottom line**: Learning JAX + GPT-OSS gives you hands-on experience with the same tools and models used by top-tier AI labs.

## Quick Start

### Local Jupyter Notebook (CPU)

```bash
# Clone and install
git clone https://github.com/atsentia/gpt-oss-jax.git
cd gpt-oss-jax
uv venv && uv pip install -e ".[jax,notebook]"

# Run notebook
jupyter lab examples/jax_inference.ipynb
```

### Google Colab (TPU)

Run on Google Cloud TPU with one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atsentia/gpt-oss-jax/blob/main/examples/jax_inference_colab_tpu.ipynb)

The TPU notebook demonstrates adaptive precision strategies:
- **TPU v2-8**: BF16 (16-bit) - ~42GB memory
- **TPU v6e**: FP8 (8-bit) - ~21GB memory

## Examples

- **[Local Notebook](examples/jax_inference.ipynb)** - CPU inference with Harmony demo
- **[Colab TPU Notebook](examples/jax_inference_colab_tpu.ipynb)** - Cloud TPU with adaptive precision

## Resources

- **Model**: [GPT-OSS-20B on HuggingFace](https://huggingface.co/openai/gpt-oss-20b)
- **Harmony Protocol**: [OpenAI Harmony](https://github.com/openai/harmony)
- **JAX Framework**: [JAX Documentation](https://jax.readthedocs.io/)

## License

Apache 2.0 - see [LICENSE](LICENSE)
