#!/usr/bin/env bash
#
#SBATCH --job-name=IMR_mem_inj_non_mem_rec
#SBATCH --output=IMR_mem_inj_non_mem_rec.log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de

srun python IMR_mem_inj_non_mem_rec.py
