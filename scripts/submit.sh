#!/usr/bin/env bash
git pull
python ../setup.py install --user
sbatch non_memory_slurm.sh
sbatch memory_slurm.sh