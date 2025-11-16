# JAX for GPT-OSS

JAX backend for GPT-OSS models with Harmony protocol support, optimized for CPU/TPU inference.

## Quick Start

```bash
# Install dependencies
uv venv
uv pip install -e ".[jax,notebook]"

# Setup checkpoint (symlink recommended)
ln -s /path/to/checkpoint weights/gpt-oss-20b

# Launch interactive notebook
source .venv/bin/activate
python -m ipykernel install --user --name=jax-for-gpt-oss
jupyter lab examples/jax_inference.ipynb
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Features

- **Fast Loading** - Orbax checkpoints load in ~5s (vs ~90s for SafeTensors)
- **Harmony Protocol** - Multi-channel reasoning with color-coded output (analysis + final answer)
- **KV Caching** - Efficient autoregressive generation
- **XLA Compilation** - Automatic optimization and kernel fusion
- **Interactive Notebook** - Progress bars, statistics, and visualization

## Harmony Protocol Demo

The [Jupyter notebook](examples/jax_inference.ipynb) includes a complete Harmony protocol demonstration with color-coded output:

- **Blue box** - Original user message
- **Green box** - Harmony-formatted prompt with special tokens
- **Yellow box** - Raw generated response
- **Green box** - Reasoning/Analysis (analysis channel)
- **Purple box** - Final Answer (final channel)

## Basic Usage

```python
from gpt_oss.jax import TokenGenerator

# Initialize generator (auto-detects Orbax or SafeTensors)
generator = TokenGenerator("weights/gpt-oss-20b", max_context_length=4096)

# Generate text
response = generator.generate(
    prompt="What is the capital of France?",
    temperature=0.0,
    max_new_tokens=50
)
print(response)
```

## Examples

- **[Interactive Notebook](examples/jax_inference.ipynb)** - Full tutorial with Harmony demo
- **[Shell Script](bin/run_harmony_example.sh)** - CLI example with colored terminal output

## Performance

| Platform | Loading | Tokens/s | Use Case |
|----------|---------|----------|----------|
| CPU (M3 Max) | ~5s | 0.5-1.0 | Development |
| TPU v4-8 | ~5s | 200-500 | Production |
| TPU v5e-8 | ~5s | 300-700 | Cost-effective |

## Requirements

- Python >= 3.12
- JAX >= 0.4.20
- openai-harmony (for Harmony protocol)

See [INSTALL.md](INSTALL.md) for complete dependency list.

## License

Apache 2.0 - see [LICENSE](LICENSE)

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [GPT-OSS-20B Model Card](https://huggingface.co/atsentia/gpt-oss-20b)
- [Harmony Protocol](https://github.com/openai/harmony)
