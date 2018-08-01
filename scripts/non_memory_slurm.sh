#!/usr/bin/env bash
#
#SBATCH --job-name=non_memory_inference
#SBATCH --output=non_memory_inference.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=500

srun python ./non_memory_injection_recovery.py