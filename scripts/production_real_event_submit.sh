#!/usr/bin/env bash
#
#SBATCH --job-name=real_event
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de

srun python production_real_event.py $1
