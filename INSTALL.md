# Installation Guide

This guide covers installation of JAX for GPT-OSS, including setup for Jupyter notebooks.

## Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Virtual environment (recommended)

## Installation Methods

### Method 1: Using uv (Recommended)

`uv` is a fast Python package installer and resolver. Install it first:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the package:

```bash
# Clone the repository (if not already done)
git clone https://github.com/atsentia/gpt-oss-jax.git
cd gpt-oss-jax

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install with JAX support
uv pip install -e ".[jax]"

# Or install with JAX + notebook support (for Jupyter)
uv pip install -e ".[jax,notebook]"

# Or install everything (dev mode)
uv pip install -e ".[dev]"
```

### Method 2: Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install with JAX support
pip install -e ".[jax]"

# Or install with JAX + notebook support
pip install -e ".[jax,notebook]"
```

## Jupyter Notebook Setup

The repository includes an interactive notebook ([`examples/jax_inference.ipynb`](examples/jax_inference.ipynb)) with:

- **Checkpoint loading** - Orbax and SafeTensors support
- **Text generation** - Progress bars, statistics, and performance tracking
- **Harmony protocol demo** - Color-coded multi-channel reasoning:
  - **Blue box** - Original user message
  - **Green box** - Harmony-formatted prompt with special tokens
  - **Yellow box** - Raw generated response (with `<|channel|>` markup)
  - **Green box** - Reasoning/Analysis (analysis channel)
  - **Purple box** - Final Answer (final channel)

### Setup Steps

1. **Install notebook dependencies** (if not already done):
   ```bash
   uv pip install -e ".[jax,notebook]"
   ```

2. **Create a Jupyter kernel** from your virtual environment:
   ```bash
   # Activate the virtual environment first
   source .venv/bin/activate

   # Create the kernel
   python -m ipykernel install --user --name=jax-for-gpt-oss --display-name "Python (jax-for-gpt-oss)"
   ```

3. **Launch Jupyter Lab**:
   ```bash
   # Make sure venv is activated
   source .venv/bin/activate

   # Launch Jupyter Lab
   jupyter lab examples/jax_inference.ipynb
   ```

4. **Select the kernel** in Jupyter Lab:
   - The notebook should automatically use the `Python (jax-for-gpt-oss)` kernel
   - If not, click the kernel name in the top right and select "Python (jax-for-gpt-oss)"

### What the Notebook Demonstrates

The final section of the notebook demonstrates the **Harmony protocol**, which enables GPT-OSS-20B to output:

1. **Reasoning** (analysis channel) - The model's internal thought process
2. **Final Answer** (final channel) - The polished response to the user

Example output for "What is the capital of France?":

- **Analysis channel**: `"We need to answer the question: 'What is the capital of France?' The answer: Paris."`
- **Final channel**: `"The capital of France is **Paris**."`

This multi-channel approach provides transparency into the model's reasoning while delivering clean final answers.

### Verify Kernel Installation

Check that the kernel is available:

```bash
jupyter kernelspec list
```

You should see:
```
jax-for-gpt-oss    /Users/yourname/Library/Jupyter/kernels/jax-for-gpt-oss
```

### Troubleshooting Jupyter Kernel

If the kernel doesn't appear or doesn't work:

1. **Verify ipykernel is installed**:
   ```bash
   source .venv/bin/activate
   python -c "import ipykernel; print('ipykernel installed')"
   ```

2. **Reinstall the kernel**:
   ```bash
   source .venv/bin/activate
   python -m ipykernel install --user --name=jax-for-gpt-oss --display-name "Python (jax-for-gpt-oss)" --force
   ```

3. **Check kernel configuration**:
   ```bash
   cat ~/Library/Jupyter/kernels/jax-for-gpt-oss/kernel.json
   ```
   
   It should point to your `.venv/bin/python`:
   ```json
   {
    "argv": [
     "/path/to/jax-for-gpt-oss/.venv/bin/python",
     ...
    ],
    ...
   }
   ```

## Dependency Groups

The package supports optional dependency groups:

- `jax` - Core JAX backend dependencies
  - `jax[cpu]>=0.4.20`
  - `jaxlib>=0.4.20`
  - `flax>=0.8.0`
  - `orbax-checkpoint>=0.5.0`
  - `safetensors>=0.5.3`
  - `numpy`
  - `tqdm`

- `notebook` - Jupyter notebook support
  - `jupyter>=1.0.0`
  - `ipywidgets>=8.0.0`

- `test` - Testing dependencies
  - `pytest>=7.4.0`
  - `pytest-xdist>=3.3.1`

- `dev` - All development dependencies (includes jax, notebook, test)

## Verify Installation

After installation, verify everything works:

```bash
# Activate virtual environment
source .venv/bin/activate

# Test imports
python -c "import jax; import flax; print(f'JAX {jax.__version__}, Flax {flax.__version__}')"

# Test gpt_oss imports
python -c "from gpt_oss.jax import TokenGenerator; from gpt_oss.jax.config import ModelConfig; print('✓ All imports successful')"
```

## Platform-Specific Notes

### macOS

- Uses CPU backend by default
- Metal (GPU) support available but not tested
- Recommended: Use `jax[cpu]` for development

### Linux

- CPU backend works out of the box
- CUDA support: Install `jax[cuda12]` or `jax[cuda11]` instead of `jax[cpu]`
- ROCm support: Install `jax[rocm]` for AMD GPUs

### Windows

- CPU backend supported
- GPU support via WSL2 recommended

### TPU (Google Cloud)

- Install TPU-specific JAX:
  ```bash
  pip install -U "jax[tpu]>=0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  ```

## Troubleshooting

### Import Errors

If you get import errors:

1. **Verify virtual environment is activated**:
   ```bash
   which python  # Should point to .venv/bin/python
   ```

2. **Reinstall in editable mode**:
   ```bash
   uv pip install -e ".[jax]" --force-reinstall
   ```

### JAX Installation Issues

If JAX fails to install:

1. **Check Python version** (must be >= 3.12):
   ```bash
   python --version
   ```

2. **Try CPU-only installation**:
   ```bash
   uv pip install "jax[cpu]>=0.4.20" "jaxlib>=0.4.20"
   ```

3. **Clear pip cache**:
   ```bash
   uv pip cache purge
   ```

### Jupyter Kernel Not Found

If Jupyter can't find the kernel:

1. **Check kernel installation**:
   ```bash
   jupyter kernelspec list
   ```

2. **Reinstall kernel** (see Jupyter Notebook Setup above)

3. **Restart Jupyter Lab**:
   ```bash
   # Kill existing Jupyter processes
   pkill -f jupyter
   
   # Restart
   jupyter lab
   ```

## Next Steps

After installation:

1. **Run the example notebook**:
   ```bash
   jupyter lab examples/jax_inference.ipynb
   ```

2. **Try the CLI**:
   ```bash
   python -m gpt_oss.generate --backend jax <checkpoint_path> -p "Your prompt"
   ```

3. **Read the documentation**:
   - [README.md](README.md) - Overview and quick start
   - [examples/jax_inference.ipynb](examples/jax_inference.ipynb) - Interactive tutorial

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/atsentia/gpt-oss-jax/issues)
- **Documentation**: See [README.md](README.md) for usage examples
- **JAX Docs**: [jax.readthedocs.io](https://jax.readthedocs.io/)
