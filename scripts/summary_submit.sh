#!/usr/bin/env bash
#
#SBATCH --job-name=summary
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de
#
#SBATCH --array=0-19
#MIN=(2000 2050 2100 2150 2200 2250 2300 2350 2400 2450 2500 2550 2600 2650 2700 2750 2800 2850 2900 2950)
#MAX=(2050 2100 2150 2200 2250 2300 2350 2400 2450 2500 2550 2600 2650 2700 2750 2800 2850 2900 2950 3000)

MIN=(0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950)
MAX=(  50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950 2000)
srun python summary.py ${MIN[$SLURM_ARRAY_TASK_ID]} ${MAX[$SLURM_ARRAY_TASK_ID]}
