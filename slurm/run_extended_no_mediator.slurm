#!/bin/bash
#SBATCH --job-name=langdist-stex-no-mediator
#SBATCH --account=def-mere
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=artifacts/logs/%x-%j.out
#SBATCH --error=artifacts/logs/%x-%j.err

set -euo pipefail

# Submit from the project root after creating artifacts/logs.
export CONFOUND_MODE="no-mediator"
export OUT_DIR="${OUT_DIR:-artifacts/stex_extended_no_mediator}"

# Reuse the shared environment setup and analysis runner.
exec bash "$SLURM_SUBMIT_DIR/slurm/run_extended.slurm"
