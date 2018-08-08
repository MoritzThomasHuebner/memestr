#!/usr/bin/env bash
#
#SBATCH --job-name=mem_only_inj
#SBATCH --output=mem_only_inj_6.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16G

srun python memory_only_injection_recovery.py
