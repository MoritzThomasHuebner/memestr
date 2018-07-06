#!/usr/bin/env bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=`
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=500

srun python ./test.py