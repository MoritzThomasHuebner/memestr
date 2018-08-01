#!/usr/bin/env bash
#
#SBATCH --job-name=test
#SBATCH --output=test_res.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000

srun python memory_injection_recovery.py
