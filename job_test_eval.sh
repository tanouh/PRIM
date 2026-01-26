#!/bin/bash
#SBATCH --job-name=siamese_test_eval
#SBATCH --output=outputs/%x_%j/log/stdout.out
#SBATCH --error=outputs/%x_%j/log/stderr.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

echo "===== SLURM test evaluation job context ====="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Started at: $(date)"
echo "=============================================="

# Ensure relative paths resolve from submit directory
cd "${SLURM_SUBMIT_DIR:-.}"

# --------------------------------------------------
# Configuration (override via env vars)
# --------------------------------------------------
ROOT_DIR="${ROOT_DIR:-data}"
CSV="${CSV:-csv/gallery_query.csv}"

MODEL_PATH="${MODEL_PATH:-outputs/siamese_train_701317/siamese.pt}"
EMBED_DIM="${EMBED_DIM:-256}"
DISTANCE="${DISTANCE:-cosine}"

BATCH_SIZE="${BATCH_SIZE:-64}"
IM_SIZE="${IM_SIZE:-256}"

THRESHOLD="${THRESHOLD:-0.7}"
SAVE_DETAILS="${SAVE_DETAILS:-false}"

# --------------------------------------------------
# Output directories
# --------------------------------------------------
OUT_DIR="${OUT_DIR:-outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR/log"

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

# ==================================================
# STEP 1: Run Prediction
# ==================================================
echo ""
echo "===== STEP 1: Running Score Prediction ====="
echo ""

PREDICT_ARGS=()
PREDICT_ARGS+=(--csv "$CSV")
PREDICT_ARGS+=(--root_dir "$ROOT_DIR")
PREDICT_ARGS+=(--model_path "$MODEL_PATH")
PREDICT_ARGS+=(--embed_dim "$EMBED_DIM")
PREDICT_ARGS+=(--distance "$DISTANCE")
PREDICT_ARGS+=(--batch_size "$BATCH_SIZE")
PREDICT_ARGS+=(--im_size "$IM_SIZE")
PREDICT_ARGS+=(--out "$OUT_DIR/scores.csv")

if [ "$SAVE_DETAILS" = "true" ]; then
  PREDICT_ARGS+=(--save_details)
fi

echo "Running: python -m scripts.predict ${PREDICT_ARGS[*]}"
srun python -u -m scripts.predict "${PREDICT_ARGS[@]}" || {
  echo "ERROR: Prediction step failed" >&2
  exit 1
}

echo "Prediction completed successfully"

# ==================================================
# STEP 2: Run Evaluation
# ==================================================
echo ""
echo "===== STEP 2: Running Evaluation ====="
echo ""

EVAL_ARGS=()
EVAL_ARGS+=(--scores_csv "$OUT_DIR/scores.csv")
EVAL_ARGS+=(--threshold "$THRESHOLD")
EVAL_ARGS+=(--out_dir "$OUT_DIR/evaluation")
EVAL_ARGS+=(--distance_type "$DISTANCE")

echo "Running: python -m scripts.evaluate_scores ${EVAL_ARGS[*]}"
python -u -m scripts.evaluate_scores "${EVAL_ARGS[@]}" || {
  echo "ERROR: Evaluation step failed" >&2
  exit 1
}

echo "Evaluation completed successfully"

# ==================================================
# Summary
# ==================================================
echo ""
echo "===== Test & Evaluation Job Summary ====="
echo "Finished at: $(date)"
echo "Results location: $OUT_DIR"
echo ""
echo "Output files:"
echo "  - Scores: $OUT_DIR/scores.csv"
echo "  - Metrics (JSON): $OUT_DIR/evaluation/metrics.json"
echo "  - Results (CSV): $OUT_DIR/evaluation/evaluation_results.csv"
echo "  - Summary (TXT): $OUT_DIR/evaluation/evaluation_summary.txt"
echo "=============================================="
