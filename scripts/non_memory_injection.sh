#!/usr/bin/env bash
#
#SBATCH --job-name=non_mem_inj_2
#SBATCH --output=non_mem_inj_2.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4000

srun python non_memory_injection_recovery.py
