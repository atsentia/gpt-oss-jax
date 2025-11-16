# Weights Directory

This directory is for storing model checkpoints. Since checkpoints are large files, they should be stored elsewhere and symlinked here.

## Setup

Create a symlink to your checkpoint directory:

```bash
# Example: Link to an Orbax checkpoint
ln -s /path/to/your/checkpoint/gpt-oss-20b weights/gpt-oss-20b

# Example: Link to a SafeTensors checkpoint
ln -s /path/to/your/safetensors/checkpoint weights/gpt-oss-20b-safetensors
```

## Usage in Notebook

The notebook uses relative paths from this directory:

```python
CHECKPOINT_PATH = "../weights/gpt-oss-20b"  # Relative to examples/
```

## Structure

```
weights/
├── README.md (this file)
├── .gitkeep (keeps directory in git)
└── gpt-oss-20b -> /path/to/actual/checkpoint (symlink)
```

## Notes

- Checkpoints are excluded from git (see `.gitignore`)
- Use descriptive symlink names (e.g., `gpt-oss-20b`, `gpt-oss-20b-safetensors`)
- The notebook will automatically detect Orbax vs SafeTensors format
