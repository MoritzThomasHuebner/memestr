#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.submit.submitter.find_unallocated_name(name=\"$LABEL\"))"
ALTERNATIVE_COMMAND="import os; outdir = ''; for i in range(0, 999): outdir = str(i).zfill(3) + "_" + \"$LABEL\"; if not os.path.exists(outdir): break; print(outdir)"
OUTDIR=`python -c "${ALTERNATIVE_COMMAND}"`
JOB_NAME="--job-name=$LABEL"
OUTPUT="--output=$OUTDIR.log"
TIME="--time=72:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=16G"
CPUS_PER_TASK="--cpus-per-task=1"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
