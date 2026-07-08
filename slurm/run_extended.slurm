#!/bin/bash
#SBATCH --job-name=langdist-stex-ext
#SBATCH --account=def-annielee
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-venv}"

STEX_BASE="${STEX_BASE:-data/stex_base.csv}"
STEX_EXTENDED="${STEX_EXTENDED:-data/stex_extended.csv}"
OUT_DIR="${OUT_DIR:-artifacts/stex_extended}"
N_SPLITS="${N_SPLITS:-5}"
RANDOM_STATE="${RANDOM_STATE:-42}"

export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts/logs "$OUT_DIR"

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi

for mod in $EXTRA_MODULES; do
  module load "$mod"
done

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "No virtualenv found at $VENV_PATH"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "which python: $(which python)"
python --version

python -c "import numpy; print('numpy ok')"
python -c "import pandas; print('pandas ok')"
python -c "import scipy; print('scipy ok')"
python -c "import sklearn; print('sklearn ok')"
python -c "import statsmodels; print('statsmodels ok')"

python extended.py \
  --stex_base "$STEX_BASE" \
  --stex_extended "$STEX_EXTENDED" \
  --outdir "$OUT_DIR" \
  --n_splits "$N_SPLITS" \
  --random_state "$RANDOM_STATE"

echo "Finished STEX extended analysis."
echo "Outputs written to $OUT_DIR"