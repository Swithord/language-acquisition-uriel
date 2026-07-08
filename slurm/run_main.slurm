#!/bin/bash
#SBATCH --job-name=langdist-core
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

CSVS="${CSVS:-data/toefl_base.csv data/stex_base.csv}"
OUT_DIR="${OUT_DIR:-artifacts/core}"
N_SPLITS="${N_SPLITS:-5}"
RANDOM_STATE="${RANDOM_STATE:-42}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"

read -r -a CSV_ARRAY <<< "$CSVS"

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
python -c "import matplotlib; print('matplotlib ok')"

FLAGS=()

if [[ "$MAKE_PLOTS" == "1" ]]; then
  FLAGS+=(--make_plots)
fi

python main.py \
  --csv "${CSV_ARRAY[@]}" \
  --outdir "$OUT_DIR" \
  --n_splits "$N_SPLITS" \
  --random_state "$RANDOM_STATE" \
  "${FLAGS[@]}"

echo "Finished core analysis."
echo "Outputs written to $OUT_DIR"