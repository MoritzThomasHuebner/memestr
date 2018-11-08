#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.submit.submitter.find_unallocated_name(name=\"$LABEL\"))"
OUTDIR=`python -c "${PYTHON_COMMAND}"`
JOB_NAME="--job-name=$LABEL"
OUTPUT=""
TIME="--time=120:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=16G"
CPUS_PER_TASK="--cpus-per-task=1"
ARRAY="--array=10,20,45,80,120,180,250,350,500"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
