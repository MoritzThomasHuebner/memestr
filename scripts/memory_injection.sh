#!/usr/bin/env bash
#
#SBATCH --job-name=mem_inj
#SBATCH --output=mem_inj_4.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=4000

srun python memory_injection_recovery.py
