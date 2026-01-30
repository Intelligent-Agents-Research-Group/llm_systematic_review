#!/usr/bin/env bash

ENV_ROOT="/blue/edorley/kyamoah/llm_systematic_review/env/llm_env/"
BIN_PATH="$ENV_ROOT/bin/python"
GROUP_ROOT="/blue/edorley/kyamoah"

# ---------------------------
# Mode detection
# ---------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # ---- Slurm job mode ----
  export PATH="$BIN_PATH:$PATH"
else
  # ---- Interactive / install mode ----
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script (do not run it)"
    exit 1
  fi

  module load conda
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_ROOT"

  module load cuda/12.8.1
  which nvcc
fi

# ---------------------------
# Caches on BLUE
# ---------------------------
mkdir -p "$GROUP_ROOT"/{hf_cache,pip_cache,torch_cache}

export HF_HOME="$GROUP_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

export PIP_USER=no
export PIP_CACHE_DIR="$GROUP_ROOT/pip_cache"

export TORCH_HOME="$GROUP_ROOT/torch_cache"
export TORCH_EXTENSIONS_DIR="$TORCH_HOME/torch_extensions"

export PYTHONNOUSERSITE=1

# ---------------------------
# Info (non-fatal)
# ---------------------------
echo "Sys Review LLM env ready"
echo "PATH python: $(command -v python 2>/dev/null || echo NOT FOUND)"
python --version 2>/dev/null || true