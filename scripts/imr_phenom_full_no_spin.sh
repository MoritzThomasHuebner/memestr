#!/usr/bin/env bash
#
#SBATCH --job-name=IMR_mem_inj_full_no_spin
#SBATCH --output=IMR_mem_inj_full_no_spin.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16G

srun python IMRPhenom_full_no_spin.py
