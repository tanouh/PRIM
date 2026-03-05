#!/bin/bash
#SBATCH --job-name=siamese_train_predict
#SBATCH --output=outputs/siamese_train_predict_%j/log/stdout.out
#SBATCH --error=outputs/siamese_train_predict_%j/log/stderr.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=30:00:00

set -euo pipefail

echo "===== SLURM combined train+predict pipeline ====="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Started at: $(date)"
echo "=================================================="

# Ensure relative paths resolve from submit directory
cd "${SLURM_SUBMIT_DIR:-.}"

# ===== CONFIGURATION =====
# Use SLURM job name and ID for output directory (consistent with job_train.sh and job_prediction.sh)
BASE_OUT_DIR="outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
TRAIN_OUT_DIR="$BASE_OUT_DIR/train_logs"
PRED_OUT_DIR="$BASE_OUT_DIR/prediction_logs"

mkdir -p "$BASE_OUT_DIR/log" "$TRAIN_OUT_DIR" "$PRED_OUT_DIR"

# Training configuration (override via env vars)
OBJECTIVE="${OBJECTIVE:-triplet}"   # contrastive | triplet
CSV_TRAIN="${CSV_TRAIN:-csv/tampar_triplets_ssl.csv}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EMBED_DIM="${EMBED_DIM:-256}"
DISTANCE="${DISTANCE:-euclidean}"          # cosine | euclidean
MARGIN="${MARGIN:-1.0}"
IM_SIZE="${IM_SIZE:-256}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PRETRAINED="${PRETRAINED:-1}"
NUM_WORKERS="${NUM_WORKERS:-}"
ROOT_DIR="${ROOT_DIR:-.}"

# Validation configuration
VAL_PAIRS_CSV_OUT="${VAL_PAIRS_CSV_OUT:-$TRAIN_OUT_DIR/val_pair_predictions.csv}"
VAL_THRESHOLD="${VAL_THRESHOLD:-0.2}"
VAL_TRIPLETS_CSV_OUT="${VAL_TRIPLETS_CSV_OUT:-$TRAIN_OUT_DIR/val_triplets_predictions.csv}"
VAL_AP_THRESHOLD="${VAL_AP_THRESHOLD:-0.2}"
VAL_AN_THRESHOLD="${VAL_AN_THRESHOLD:-0.2}"
VAL_DELTA_THRESHOLD="${VAL_DELTA_THRESHOLD:-0.2}"

# Prediction configuration (override via env vars)
CSV_PREDICT="${CSV_PREDICT:-csv/gallery_query.csv}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-64}"
SAVE_DETAILS="${SAVE_DETAILS:-false}"
THRESHOLD="${THRESHOLD:-0.2}"

# Model path (trained model will be saved here)
TRAINED_MODEL="$TRAIN_OUT_DIR/siamese.pt"

echo "Configuration:"
echo "  BASE_OUT_DIR: $BASE_OUT_DIR"
echo "  TRAIN_OUT_DIR: $TRAIN_OUT_DIR"
echo "  PRED_OUT_DIR: $PRED_OUT_DIR"
echo "  TRAINED_MODEL: $TRAINED_MODEL"
echo ""

# Activate conda environment if specified
CONDA_ENV="${CONDA_ENV:-cuda118-gpu}"
if [ -n "$CONDA_ENV" ]; then
  set +u
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
  set -u
fi

# Diagnostics
nvidia-smi || true
python -V || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ===== TRAINING PHASE =====
echo ""
echo "=========================================="
echo "STARTING TRAINING PHASE"
echo "=========================================="
echo "Started at: $(date)"

TRAIN_ARGS=()
TRAIN_ARGS+=(--objective "$OBJECTIVE")

for csv_path in $CSV_TRAIN; do
  TRAIN_ARGS+=(--csv "$csv_path")
done

TRAIN_ARGS+=(--root_dir "$ROOT_DIR")
TRAIN_ARGS+=(--epochs "$EPOCHS")
TRAIN_ARGS+=(--batch_size "$BATCH_SIZE")
TRAIN_ARGS+=(--lr "$LR")
TRAIN_ARGS+=(--weight_decay "$WEIGHT_DECAY")
TRAIN_ARGS+=(--embed_dim "$EMBED_DIM")
TRAIN_ARGS+=(--distance "$DISTANCE")
TRAIN_ARGS+=(--margin "$MARGIN")
TRAIN_ARGS+=(--im_size "$IM_SIZE")
TRAIN_ARGS+=(--pin_memory "$PIN_MEMORY")
TRAIN_ARGS+=(--pretrained "$PRETRAINED")
TRAIN_ARGS+=(--save_path "$TRAINED_MODEL")
TRAIN_ARGS+=(--sbatch 1)

if [ -n "$VAL_PAIRS_CSV_OUT" ]; then
  TRAIN_ARGS+=(--val_pairs_csv_out "$VAL_PAIRS_CSV_OUT")
fi
if [ -n "$VAL_THRESHOLD" ]; then
  TRAIN_ARGS+=(--val_threshold "$VAL_THRESHOLD")
fi
if [ -n "$VAL_TRIPLETS_CSV_OUT" ]; then
  TRAIN_ARGS+=(--val_triplets_csv_out "$VAL_TRIPLETS_CSV_OUT")
fi
if [ -n "$VAL_AP_THRESHOLD" ]; then
  TRAIN_ARGS+=(--val_ap_threshold "$VAL_AP_THRESHOLD")
fi
if [ -n "$VAL_AN_THRESHOLD" ]; then
  TRAIN_ARGS+=(--val_an_threshold "$VAL_AN_THRESHOLD")
fi
if [ -n "$VAL_DELTA_THRESHOLD" ]; then
  TRAIN_ARGS+=(--val_delta_threshold "$VAL_DELTA_THRESHOLD")
fi

if [ -n "$NUM_WORKERS" ]; then
  TRAIN_ARGS+=(--num_workers "$NUM_WORKERS")
fi

echo "Running: python -m scripts.train ${TRAIN_ARGS[*]}"
srun python -u -m scripts.train "${TRAIN_ARGS[@]}" 2>&1 | tee "$TRAIN_OUT_DIR/train.log"

if [ ! -f "$TRAINED_MODEL" ]; then
  echo "ERROR: Training failed - model not saved at $TRAINED_MODEL" >&2
  exit 1
fi

echo "Training completed at: $(date)"
echo "Model saved to: $TRAINED_MODEL"
echo ""

# ===== PREDICTION PHASE =====
echo "=========================================="
echo "STARTING PREDICTION PHASE"
echo "=========================================="
echo "Started at: $(date)"

PRED_ARGS=()
PRED_ARGS+=(--csv "$CSV_PREDICT")
PRED_ARGS+=(--root_dir "$ROOT_DIR")
PRED_ARGS+=(--model_path "$TRAINED_MODEL")
PRED_ARGS+=(--embed_dim "$EMBED_DIM")
PRED_ARGS+=(--distance "$DISTANCE")
PRED_ARGS+=(--batch_size "$PRED_BATCH_SIZE")
PRED_ARGS+=(--im_size "$IM_SIZE")
PRED_ARGS+=(--out "$PRED_OUT_DIR/scores.csv")

if [ -n "$NUM_WORKERS" ]; then
  PRED_ARGS+=(--num_workers "$NUM_WORKERS")
fi

if [ "$SAVE_DETAILS" = "true" ]; then
  PRED_ARGS+=(--save_details)
fi

echo "Running: python -m scripts.predict ${PRED_ARGS[*]}"
srun python -u -m scripts.predict "${PRED_ARGS[@]}" 2>&1 | tee "$PRED_OUT_DIR/predict.log"

echo "Prediction completed at: $(date)"
echo "Results saved to: $PRED_OUT_DIR"
echo ""

# ===== CLEANUP =====
echo "Removing trained model to save disk space..."
rm -f "$TRAINED_MODEL"
echo "Model removed: $TRAINED_MODEL"
echo ""

# ===== SUMMARY =====
echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "Base output directory: $BASE_OUT_DIR"
echo "Training logs: $TRAIN_OUT_DIR"
echo "Trained model: (deleted after prediction)"
echo "Prediction logs: $PRED_OUT_DIR"
echo "Prediction scores: $PRED_OUT_DIR/scores.csv"
echo "SLURM logs: $BASE_OUT_DIR/log/"
echo "Finished at: $(date)"
echo "=========================================="
