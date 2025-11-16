#!/bin/bash
# JAX inference wrapper
set -e

CHECKPOINT=${1:-"../gpt-oss-20b/original/"}
PROMPT=${2:-"Who wrote Romeo and Juliet?"}
MAX_TOKENS=${3:-20}

export XLA_FLAGS="--xla_cpu_enable_fast_math=true --xla_cpu_fast_math_honor_infs=true --xla_cpu_fast_math_honor_nans=true"
export JAX_COMPILATION_CACHE_DIR=~/.cache/jax

python -m gpt_oss.generate --backend jax "$CHECKPOINT" -p "$PROMPT" -l "$MAX_TOKENS" -t 0.0
