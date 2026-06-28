#!/bin/bash
#SBATCH --job-name=iwarpgan-s1
#SBATCH --partition=a100
#SBATCH --nodelist=ngpu06
#SBATCH --error=/userspace/sse/iris2026/logs/stage1.err
#SBATCH --output=/userspace/sse/iris2026/logs/stage1.log
#SBATCH --open-mode=append

# ========================= НАСТРОЙКИ =========================
BATCH=128
BATCH_GPU=128
KIMG=25000
CONDA_ENV=/userspace/sse/.conda/envs/warp
REPO=/userspace/sse/iris2026/iWarpGAN
DATA=/userspace/sse/Datasets/casia_ours_128.zip
OUTDIR=/userspace/sse/iris2026/training-runs
# ==============================================================

export TORCH_EXTENSIONS_DIR=/userspace/sse/.cache/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

mkdir -p "$OUTDIR" "$(dirname "$OUTDIR")/logs"

source /userspace/sse/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Load nvcc if available via modules (harmless if module doesn't exist)
module load cuda/12.1 2>/dev/null || true

cd "$REPO" || exit 1

python -m train \
    --cfg=stylegan2 \
    --data="$DATA" \
    --outdir="$OUTDIR" \
    --gpus=1 \
    --batch="$BATCH" \
    --batch-gpu="$BATCH_GPU" \
    --cond=True \
    --gamma=8.2 \
    --mbstd-group="$(( BATCH_GPU > 1 ? BATCH_GPU : 1 ))" \
    --metrics=none \
    --kimg="$KIMG"
