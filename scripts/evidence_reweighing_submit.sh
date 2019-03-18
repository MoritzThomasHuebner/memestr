#!/usr/bin/env bash
#
#SBATCH --job-name=evidence_reweighing
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de

srun python evidence_recalculation.py $1
