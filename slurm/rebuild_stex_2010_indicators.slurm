#!/bin/bash
#SBATCH --job-name=stex-2010-indicators
#SBATCH --account=def-mere
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON_MODULE="${PYTHON_MODULE:-python}"
EXTRA_MODULES="${EXTRA_MODULES:-}"
VENV_PATH="${VENV_PATH:-venv}"

SCRIPT="${SCRIPT:-data/unprocessed_indicators/rebuild_stex_2010_indicators.py}"
DATA_DIR="${DATA_DIR:-data}"
HDI_PATH="${HDI_PATH:-data/unprocessed_indicators/human-development-index.csv}"
WDI_PATH="${WDI_PATH:-data/unprocessed_indicators/wdi_data.csv}"
TARGET_YEAR="${TARGET_YEAR:-2010}"
MAKE_BACKUPS="${MAKE_BACKUPS:-1}"

mkdir -p artifacts/logs

if [[ -n "$PYTHON_MODULE" ]]; then
  module load "$PYTHON_MODULE"
fi

for mod in $EXTRA_MODULES; do
  module load "$mod"
done

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
else
  echo "No virtualenv found at $VENV_PATH" >&2
  exit 1
fi

export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "which python: $(which python)"
python --version

test -f "$SCRIPT"
test -f "$DATA_DIR/stex_extended.csv"
test -f "$HDI_PATH"
test -f "$WDI_PATH"

python -c "import numpy; print('numpy ok')"
python -c "import pandas; print('pandas ok')"

FLAGS=()
if [[ "$MAKE_BACKUPS" != "1" ]]; then
  FLAGS+=(--no-backup)
fi

python "$SCRIPT" \
  --data-dir "$DATA_DIR" \
  --hdi-path "$HDI_PATH" \
  --wdi-path "$WDI_PATH" \
  --target-year "$TARGET_YEAR" \
  "${FLAGS[@]}"

echo "Finished rebuilding STEX country indicators."
echo "Outputs:"
echo "  $DATA_DIR/stex_extended.csv"
echo "  $DATA_DIR/stex_confounder_metadata.csv"
