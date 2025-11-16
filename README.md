# JAX for GPT-OSS

JAX backend for GPT-OSS models, optimized for CPU/TPU inference.

## Features

- **Fast Loading:** Orbax checkpoints load 18x faster than SafeTensors (~5s vs ~90s)
- **Efficient Generation:** KV caching for O(n) complexity
- **Cross-Platform:** Works on CPU, GPU (CUDA/ROCm/Metal), and TPU
- **XLA Compilation:** Automatic optimization and kernel fusion
- **Jupyter Notebooks:** Interactive examples with progress bars and statistics

## Quick Start

### Installation

```bash
# Create virtual environment (using uv)
uv venv

# Install with JAX support
uv pip install -e ".[jax]"

# Or install with JAX + notebook support
uv pip install -e ".[jax,notebook]"
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

### Basic Usage

```python
from gpt_oss.jax import TokenGenerator
import jax

# Initialize generator (auto-detects Orbax or SafeTensors)
generator = TokenGenerator("path/to/checkpoint", max_context_length=4096)

# Generate tokens
prompt_tokens = [1, 2, 3]  # Your prompt tokens
for token, logprob in generator.generate(prompt_tokens, max_tokens=50, temperature=0.8):
    print(f"Token: {token}")
```

### Jupyter Notebook

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Jupyter Lab
jupyter lab examples/jax_inference.ipynb
```

The notebook includes:
- Checkpoint loading (Orbax/SafeTensors)
- Model initialization
- Text generation with progress bars
- Performance statistics (TTFT, tokens/second)
- Temperature sampling examples

## Checkpoints

### Setup Checkpoint Directory

**Recommended**: Create a symlink in the `weights/` directory:

```bash
# From the repository root
ln -s /path/to/your/checkpoint weights/gpt-oss-20b
```

This allows using relative paths in notebooks and scripts: `weights/gpt-oss-20b`

See [weights/README.md](weights/README.md) for more details.

### Orbax Format (Recommended)

Fast loading (~5 seconds):

```bash
# Use Orbax checkpoint
python -m gpt_oss.generate --backend jax weights/gpt-oss-20b "Your prompt"
```

### SafeTensors Format

Slower loading (~90 seconds) but cross-framework compatible:

```bash
python -m gpt_oss.generate --backend jax weights/gpt-oss-20b-safetensors "Your prompt"
```

## Performance

| Platform | Loading Time | Tokens/Second | Notes |
|----------|--------------|---------------|-------|
| CPU (M3 Max) | ~5s (Orbax) | 10-20 | Good for development |
| TPU v4-8 | ~5s (Orbax) | 200-500 | Production inference |
| TPU v5e-8 | ~5s (Orbax) | 300-700 | Cost-effective |
| TPU v6e (Trillium) | ~5s (Orbax) | 500-1000+ | Latest hardware |

## Examples

- [Local Jupyter Notebook](examples/jax_inference.ipynb) - Interactive inference tutorial
- [CLI Script](bin/run_jax_inference.sh) - Command-line inference example

## Requirements

- Python >= 3.12
- JAX >= 0.4.20
- Flax >= 0.8.0
- See [INSTALL.md](INSTALL.md) for full dependency list

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/jax -v

# Skip slow tests
pytest tests/jax -m "not slow" -v
```

## License

Apache 2.0 - see [LICENSE](LICENSE) file.

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [GPT-OSS-20B Model Card](https://huggingface.co/atsentia/gpt-oss-20b)
