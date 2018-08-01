#!/usr/bin/env bash
#
#SBATCH --job-name=non_mem_inj
#SBATCH --output=non_mem_inj.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000

srun python non_memory_injection_recovery.py
