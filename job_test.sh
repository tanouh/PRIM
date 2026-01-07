#!/bin/bash
#SBATCH --job-name=siamese_test        # Job name
#SBATCH --output=outputs/%x_%j/log/stdout.out             # Stdout
#SBATCH --error=outputs/%x_%j/log/stderr.err              # Stderr
#SBATCH --partition=P100               # GPU partition (e.g., A100, V100, P100)
#SBATCH --gres=gpu:1                   # Request 1 GPU (optional, set 0 if CPU-only)
#SBATCH --cpus-per-task=4              # CPU cores per task
#SBATCH --mem=4G                       # System RAM
#SBATCH --time=02:00:00                # Walltime (hh:mm:ss)

set -euo pipefail

echo "===== SLURM test job context ====="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Started at: $(date)"
echo "=================================="

# Ensure relative paths resolve from submit directory
cd "${SLURM_SUBMIT_DIR:-.}"

# Configuration (override by exporting env vars before sbatch)
# Example: OBJECTIVE=contrastive MODEL_PATH=outputs/.../siamese.pt sbatch job_test.sh
OBJECTIVE="${OBJECTIVE:-triplet}"   # contrastive | triplet
ROOT_DIR="${ROOT_DIR:-.}"

# Default CSV if not provided
if [ -z "${CSV:-}" ]; then
  if [ "$OBJECTIVE" = "contrastive" ]; then
    CSV="csv/tampar_pairs_test.csv"
  else
    CSV="csv/drive_triplets_test.csv"
  fi
fi

MODEL_PATH="${MODEL_PATH:-outputs/siamese_train_701317/siamese.pt}"   # required; override this
BATCH_SIZE="${BATCH_SIZE:-32}"
IM_SIZE="${IM_SIZE:-256}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PRETRAINED="${PRETRAINED:-1}"
NUM_WORKERS="${NUM_WORKERS:-}"  # optional manual override

# Outputs
OUT_DIR="${OUT_DIR:-outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR/log"

# Validation outputs (contrastive)
VAL_PAIRS_CSV_OUT="${VAL_PAIRS_CSV_OUT:-$OUT_DIR/test_pair_predictions.csv}"
VAL_THRESHOLD="${VAL_THRESHOLD:-}"

# Validation outputs (triplet)
VAL_TRIPLETS_CSV_OUT="${VAL_TRIPLETS_CSV_OUT:-$OUT_DIR/test_triplets_predictions.csv}"
VAL_AP_THRESHOLD="${VAL_AP_THRESHOLD:-}"
VAL_AN_THRESHOLD="${VAL_AN_THRESHOLD:-}"
VAL_DELTA_THRESHOLD="${VAL_DELTA_THRESHOLD:-}"

# Conda environment (optional)
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
    conda activate "$CONDA_ENV" || echo "[WARN] Failed to activate conda env '$CONDA_ENV', proceeding with system python"
  else
    echo "[WARN] conda not found; proceeding with system python"
  fi
fi

# Diagnostics
nvidia-smi || true
python -V || true

# Avoid CPU oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Build argument list for scripts/test.py
ARGS=()
ARGS+=(--objective "$OBJECTIVE")

# --csv supports multiple files (nargs '+'): split CSV on whitespace
for csv_path in $CSV; do
  ARGS+=(--csv "$csv_path")
done

ARGS+=(--root_dir "$ROOT_DIR")
ARGS+=(--model_path "$MODEL_PATH")
ARGS+=(--batch_size "$BATCH_SIZE")
ARGS+=(--im_size "$IM_SIZE")
ARGS+=(--pin_memory "$PIN_MEMORY")
ARGS+=(--pretrained "$PRETRAINED")

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

# Require model path
if [ -z "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not set. Set env var MODEL_PATH to the saved model file." >&2
  exit 2
fi

echo "Running test: scripts/test.py with args: ${ARGS[*]}"

# Run
srun python -u -m scripts.test "${ARGS[@]}"

echo "Test job finished at: $(date)"
echo "Outputs (if any) saved to: $OUT_DIR"
