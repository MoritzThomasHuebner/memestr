#!/usr/bin/env bash
#
#SBATCH --job-name=evidence_reweighing
#SBATCH --output=evidence_reweighing.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00
#SBATCH --mem-per-cpu=1G

srun python evidence_recalculation.py $1 $2 $3
