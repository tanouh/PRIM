#!/bin/bash
#SBATCH --job-name=siamese_train       # Job name
#SBATCH --output=outputs/%x_%j/log/stdout.out             # Stdout (%x=job-name, %j=job-id)
#SBATCH --error=outputs/%x_%j/log/stderr.err              # Stderr
#SBATCH --partition=P100               # GPU partition (e.g., A100, V100, P100)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=8              # CPU cores per task
#SBATCH --mem=10G                      # System RAM
#SBATCH --time=24:00:00                # Walltime (hh:mm:ss)

set -euo pipefail

echo "===== SLURM context ====="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Started at: $(date)"
echo "========================="

# Ensure relative paths resolve from submit directory
cd "${SLURM_SUBMIT_DIR:-.}"

# Configuration (override by exporting env vars before sbatch)
# Example: OBJECTIVE=triplet EPOCHS=50 BATCH_SIZE=64 LR=3e-4 sbatch job_train.sh
OBJECTIVE="${OBJECTIVE:-triplet}"   # contrastive | triplet
ROOT_DIR="${ROOT_DIR:-.}"

# Choose default CSV based on objective if not provided
if [ -z "${CSV:-}" ]; then
  if [ "$OBJECTIVE" = "contrastive" ]; then
    CSV="csv/tampar_pairs.csv"
  else
    CSV="csv/tampar_triplets.csv"
  fi
fi

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EMBED_DIM="${EMBED_DIM:-256}"
DISTANCE="${DISTANCE:-euclidean}"          # cosine | euclidean
MARGIN="${MARGIN:-1.0}"
IM_SIZE="${IM_SIZE:-256}"
PIN_MEMORY="${PIN_MEMORY:-1}"           # 1=True, 0=False
PRETRAINED="${PRETRAINED:-1}"           # 1=True, 0=False
NUM_WORKERS="${NUM_WORKERS:-}"          # optional manual override; otherwise auto from SLURM

# Outputs
OUT_DIR="${OUT_DIR:-outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"
SAVE_PATH="${SAVE_PATH:-$OUT_DIR/siamese.pt}"

# Validation outputs (for contrastive objective)
# Can be overridden via environment variables:
#   VAL_PAIRS_CSV_OUT=/path/to/file.csv
#   VAL_THRESHOLD=0.5
VAL_PAIRS_CSV_OUT="${VAL_PAIRS_CSV_OUT:-$OUT_DIR/val_pair_predictions.csv}"
VAL_THRESHOLD="${VAL_THRESHOLD:-0.2}"

# Validation outputs (for triplet objective)
# Can be overridden via environment variables:
#   VAL_TRIPLETS_CSV_OUT=/path/to/file.csv
#   VAL_AP_THRESHOLD=0.5
#   VAL_AN_THRESHOLD=0.5
#   VAL_DELTA_THRESHOLD=0.2
VAL_TRIPLETS_CSV_OUT="${VAL_TRIPLETS_CSV_OUT:-$OUT_DIR/val_triplets_predictions.csv}"
VAL_AP_THRESHOLD="${VAL_AP_THRESHOLD:-0.2}"
VAL_AN_THRESHOLD="${VAL_AN_THRESHOLD:-}"
VAL_DELTA_THRESHOLD="${VAL_DELTA_THRESHOLD:-}"

# Activate conda environment (default to 'cuda118' if CONDA_ENV not set)
CONDA_ENV="${CONDA_ENV:-cuda118-gpu}"

if [ -n "$CONDA_ENV" ]; then
  # Try common conda init paths
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
    conda activate "$CONDA_ENV" || echo "[WARN] Failed to activate conda env '$CONDA_ENV', proceeding with system python"
  else
    echo "[WARN] conda not found; cannot activate env '$CONDA_ENV'"
  fi
fi

# Diagnostics
nvidia-smi || true
python -V || true

# Avoid CPU oversubscription in dataloaders and BLAS
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Build argument list for scripts/train.py
ARGS=()
ARGS+=(--objective "$OBJECTIVE")

# --csv supports multiple files (nargs '+'): split CSV on whitespace
for csv_path in $CSV; do
  ARGS+=(--csv "$csv_path")
done

ARGS+=(--root_dir "$ROOT_DIR")
ARGS+=(--epochs "$EPOCHS")
ARGS+=(--batch_size "$BATCH_SIZE")
ARGS+=(--lr "$LR")
ARGS+=(--weight_decay "$WEIGHT_DECAY")
ARGS+=(--embed_dim "$EMBED_DIM")
ARGS+=(--distance "$DISTANCE")
ARGS+=(--margin "$MARGIN")
ARGS+=(--im_size "$IM_SIZE")
ARGS+=(--pin_memory "$PIN_MEMORY")
ARGS+=(--pretrained "$PRETRAINED")
ARGS+=(--save_path "$SAVE_PATH")
ARGS+=(--sbatch 1)

# Validation logging (contrastive only; harmless if unused)
if [ -n "$VAL_PAIRS_CSV_OUT" ]; then
  ARGS+=(--val_pairs_csv_out "$VAL_PAIRS_CSV_OUT")
fi
if [ -n "$VAL_THRESHOLD" ]; then
  ARGS+=(--val_threshold "$VAL_THRESHOLD")
fi

# Validation logging (triplet only; harmless if unused)
if [ -n "$VAL_TRIPLETS_CSV_OUT" ]; then
  ARGS+=(--val_triplets_csv_out "$VAL_TRIPLETS_CSV_OUT")
fi
if [ -n "$VAL_AP_THRESHOLD" ]; then
  ARGS+=(--val_ap_threshold "$VAL_AP_THRESHOLD")
fi
if [ -n "$VAL_AN_THRESHOLD" ]; then
  ARGS+=(--val_an_threshold "$VAL_AN_THRESHOLD")
fi
if [ -n "$VAL_DELTA_THRESHOLD" ]; then
  ARGS+=(--val_delta_threshold "$VAL_DELTA_THRESHOLD")
fi

# Allow manual override of num_workers if explicitly provided
if [ -n "$NUM_WORKERS" ]; then
  ARGS+=(--num_workers "$NUM_WORKERS")
fi

echo "Running: scripts/train.py with args: ${ARGS[*]}"
srun python -u -m scripts.train "${ARGS[@]}"

echo "Job finished at: $(date)"
echo "Model saved to: $SAVE_PATH"
