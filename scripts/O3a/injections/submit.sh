#!/bin/bash
#
#SBATCH --job-name=inj
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G

srun python run_injection.py ${1} ${2}