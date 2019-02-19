#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.core.submit.find_unallocated_name(name=\"$LABEL\"))"
OUTDIR=`python -c "${PYTHON_COMMAND}"`
mkdir ${OUTDIR}
JOB_NAME="--job-name=$LABEL"
OUTPUT=""
TIME="--time=1:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=4G"
CPUS_PER_TASK="--cpus-per-task=8"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
ARRAY=""
