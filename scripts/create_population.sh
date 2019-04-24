#!/usr/bin/env bash
#
#SBATCH --job-name=pop
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de

srun python create_population.py $1 $2