#!/bin/bash
#SBATCH --account=aip-agoldenb
#SBATCH --job-name=uriel-analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_outputs/%x-%j.out
#SBATCH --error=slurm_outputs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p slurm_outputs artifacts rlib

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export FLEXIBLAS_NUM_THREADS=1
export R_MAX_VSIZE=60GB

export R_LIBS_USER="$SLURM_SUBMIT_DIR/rlib"
export OUTPUT_DIR="$SLURM_SUBMIT_DIR/artifacts/results_uriel"
export DATA_DIR="$SLURM_SUBMIT_DIR/data"

export GLMNET_INNER_FOLDS=3
export GLMNET_NLAMBDA=30
export STEX_OUTER_V=5
export STEX_OUTER_REPEATS=3

module --force purge
module load StdEnv/2023
module load gcc/14.3 r/4.5.0
module list

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

Rscript --vanilla "${R_FILE}"