#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


#srun python reweight.py ${1} ${2} 16 ${3}
srun python reweight_memory_amplitude.py ${1} ${2} 16 ${3}