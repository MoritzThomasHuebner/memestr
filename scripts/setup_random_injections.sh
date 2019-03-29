#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.core.submit.find_unallocated_name(name=\"$LABEL\"))"
OUTDIR="${2}"
#OUTDIR=`python -c "${PYTHON_COMMAND}"`
mkdir - ${OUTDIR}
JOB_NAME="--job-name=$LABEL"
OUTPUT=""
TIME="--time=168:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=1G"
CPUS_PER_TASK="--cpus-per-task=2"
ARRAY="--array=0-7"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"