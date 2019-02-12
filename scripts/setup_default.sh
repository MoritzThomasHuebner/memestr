#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.core.submit.find_unallocated_name(name=\"$LABEL\"))"
OUTDIR=`python -c "${PYTHON_COMMAND}"`
JOB_NAME="--job-name=$LABEL"
OUTPUT="--output=$OUTDIR.log"
TIME="--time=72:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=4G"
CPUS_PER_TASK="--cpus-per-task=1"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
