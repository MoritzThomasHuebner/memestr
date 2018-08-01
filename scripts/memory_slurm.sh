#!/usr/bin/env bash
#
#SBATCH --job-name=memory_inference
#SBATCH --output=memory_inference.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=500

srun python ./memory_injection_recovery.py