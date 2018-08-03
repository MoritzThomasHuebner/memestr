#!/usr/bin/env bash
PYTHON_FILE=$1
JOB_NAME=${PYTHON_FILE}
OUTPUT=$2
#
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT}
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16g

srun python ${PYTHON_FILE}
