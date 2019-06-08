#!/usr/bin/env bash
#
#SBATCH --job-name=real_event
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de
#
#SBATCH --array=0-63

srun python production_real_event.py $1 64 ${SLURM_ARRAY_TASK_ID}
