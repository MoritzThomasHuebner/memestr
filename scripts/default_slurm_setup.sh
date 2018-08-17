#!/usr/bin/env bash

LABEL="${1::-3}"
echo ${LABEL}
PYTHON_COMMAND="import memestr.submit.submitter; find_unallocated_name(name=\"$LABEL\")"
echo ${PYTHON_COMMAND}
python -c "${PYTHON_COMMAND}"
echo ${LABEL}
JOB_NAME="--job_name=$LABEL"
OUTPUT="--output=$LABEL.log"
TIME="--time=72:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=16G"
CPUS_PER_TASK="--cpus-per-task=1"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
