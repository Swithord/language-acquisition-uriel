#!/bin/bash
#SBATCH --account=aip-agoldenb
#SBATCH --job-name=uriel-analysis
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_outputs/%x-%j.out
#SBATCH --error=slurm_outputs/%x-%j.err

set -euo pipefail

# working directories

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_outputs artifacts rlib

# thread and memory controls

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export FLEXIBLAS_NUM_THREADS=1
export R_MAX_VSIZE=60GB

# R library and project paths

export R_LIBS_USER="$SLURM_SUBMIT_DIR/rlib"
export OUTPUT_DIR="$SLURM_SUBMIT_DIR/artifacts/results_uriel"
export DATA_DIR="$SLURM_SUBMIT_DIR/data"

# glmnet controls

export GLMNET_INNER_FOLDS=3
export GLMNET_NLAMBDA=30

# currently unused by analysis.R, but kept here in case you later wire them in

export STEX_OUTER_V=5
export STEX_OUTER_REPEATS=3

# outlier and leverage filtering controls
# valid modes: none, zscore, leverage, both
# valid scopes: none, toefl, stex, both
# if OUTLIER_USE_SELECTED_ONLY=true, filtering is based on:
# z_syntactic_dist, z_morphological_dist, z_phonological_dist, z_inventory_dist
# otherwise it uses all 8 standardized distance variables

export OUTLIER_FILTER_MODE=none
export OUTLIER_FILTER_SCOPE=none
export OUTLIER_Z_THRESHOLD=2.5
export OUTLIER_LEVERAGE_MULT=2.5
export OUTLIER_USE_SELECTED_ONLY=true

# module setup

module --force purge
module load StdEnv/2023
module load gcc/14.3 r/4.5.0
module list

# package installation into user library if needed

Rscript --vanilla - <<'RSCRIPT'
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)

locks <- list.files(lib, pattern = "^00LOCK", full.names = TRUE)
if (length(locks)) unlink(locks, recursive = TRUE, force = TRUE)

repos <- "https://cloud.r-project.org"

pkgs <- c(
  "data.table",
  "ggplot2",
  "fixest",
  "broom",
  "glmnet",
  "Matrix",
  "patchwork",
  "scales"
)

need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE, lib.loc = lib)]
if (length(need)) install.packages(need, lib = lib, repos = repos)

q(status = 0)
RSCRIPT

# run analysis

R_FILE="analysis.R"
if [[ ! -f "${R_FILE}" ]]; then
  echo "ERROR: Cannot find ${R_FILE} in ${PWD}" >&2
  exit 1
fi

echo "running ${R_FILE}"
echo "submit dir: ${SLURM_SUBMIT_DIR}"
echo "output dir: ${OUTPUT_DIR}"
echo "data dir:   ${DATA_DIR}"
echo "cpus:       ${SLURM_CPUS_PER_TASK}"
echo "filter mode: ${OUTLIER_FILTER_MODE}"
echo "filter scope: ${OUTLIER_FILTER_SCOPE}"
echo "z threshold: ${OUTLIER_Z_THRESHOLD}"
echo "leverage mult: ${OUTLIER_LEVERAGE_MULT}"
echo "selected vars only: ${OUTLIER_USE_SELECTED_ONLY}"

Rscript --vanilla "${R_FILE}"