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
ARRAY="--array=0-7"
NOISE_SEEDS={36380 66957 81888 74796 60079 70495 19376 76630}
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
