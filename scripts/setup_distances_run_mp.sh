#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.submit.submitter.find_unallocated_name(name=\"$LABEL\"))"
OUTDIR=`python -c "${PYTHON_COMMAND}"`
JOB_NAME="--job-name=$LABEL"
OUTPUT=""
TIME="--time=1:00:00"
NUMBER_OF_TASKS=4
NTASKS="--ntasks="${NUMBER_OF_TASKS}"
MEM_PER_CPU="--mem-per-cpu=8G"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
