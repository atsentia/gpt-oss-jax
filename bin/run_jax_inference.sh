#!/bin/bash
# Convenience script for running JAX inference
# Usage: ./run_jax_inference.sh

python -m gpt_oss.generate \
    --backend jax \
    ../atsentia-gpt-oss-experiments/gpt-oss-jax-orbax/orbax_checkpoints/gpt-oss-20b \
    -p "Who wrote Romeo and Juliet?" \
    -l 13 \
    -t 0.0
