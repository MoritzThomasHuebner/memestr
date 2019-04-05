#!/usr/bin/env bash
#SBATCH job-name="test"
#SBATCH time=1:00:00
#SBATCH ntasks=1
#SBATCH mem-per-cpu=1G
#SBATCH cpus-per-task=4
python debug_script.py