#!/bin/bash
#SBATCH --job-name=siamese_infer
#SBATCH --output=outputs/%x_%j/log/stdout.out
#SBATCH --error=outputs/%x_%j/log/stderr.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

echo "===== SLURM inference job context ====="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Started at: $(date)"
echo "======================================"

# Ensure relative paths resolve from submit directory
cd "${SLURM_SUBMIT_DIR:-.}"

# --------------------------------------------------
# Configuration (override via env vars)
# --------------------------------------------------
ROOT_DIR="${ROOT_DIR:-data}"
CSV="${CSV:-csv/gallery_query.csv}"

MODEL_PATH="${MODEL_PATH:-outputs/siamese_train_669423/siamese.pt}"
EMBED_DIM="${EMBED_DIM:-256}"
DISTANCE="${DISTANCE:-cosine}"

BATCH_SIZE="${BATCH_SIZE:-64}"
IM_SIZE="${IM_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PRETRAINED="${PRETRAINED:-1}"

TOPK="${TOPK:-10}"

# --------------------------------------------------
# Output directories
# --------------------------------------------------
OUT_DIR="${OUT_DIR:-outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR/log"

OUT_CSV="${OUT_CSV:-$OUT_DIR/inference_results.csv}"
OUT_EMB_DIR="${OUT_EMB_DIR:-$OUT_DIR/embeddings}"

# --------------------------------------------------
# Conda environment
# --------------------------------------------------
CONDA_ENV="${CONDA_ENV:-cuda118}"
if [ -n "$CONDA_ENV" ]; then
  for script in \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$script" ]; then
      source "$script"
      break
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV" || echo "[WARN] Failed to activate conda env"
  else
    echo "[WARN] conda not found; using system python"
  fi
fi

# --------------------------------------------------
# Diagnostics
# --------------------------------------------------
nvidia-smi || true
python -V || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --------------------------------------------------
# Build argument list
# --------------------------------------------------
ARGS=()
ARGS+=(--csv "$CSV")
ARGS+=(--root_dir "$ROOT_DIR")
ARGS+=(--model_path "$MODEL_PATH")
ARGS+=(--embed_dim "$EMBED_DIM")
ARGS+=(--distance "$DISTANCE")
ARGS+=(--batch_size "$BATCH_SIZE")
ARGS+=(--im_size "$IM_SIZE")
ARGS+=(--out "$OUT_DIR")
# ARGS+=(--out_csv "$OUT_CSV")
# ARGS+=(--out_embeddings_dir "$OUT_EMB_DIR")

if [ -n "$NUM_WORKERS" ]; then
  ARGS+=(--num_workers "$NUM_WORKERS")
fi

# --------------------------------------------------
# Safety checks
# --------------------------------------------------
if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
  exit 2
fi

if [ ! -f "$CSV" ]; then
  echo "ERROR: CSV not found: $CSV" >&2
  exit 2
fi

echo "Running inference with args:"
echo "  python -m scripts.infer ${ARGS[*]}"

# --------------------------------------------------
# Run inference
# --------------------------------------------------
srun python -u -m scripts.infer "${ARGS[@]}"

echo "Inference job finished at: $(date)"
echo "Results saved to: $OUT_DIR"
