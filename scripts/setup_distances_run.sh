#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.submit.submitter.find_unallocated_name(name=\"$LABEL\"))"
OUTDIR="003_$LABEL"#`python -c "${PYTHON_COMMAND}"`
JOB_NAME="--job-name=$LABEL"
OUTPUT=""
TIME="--time=120:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=16G"
CPUS_PER_TASK="--cpus-per-task=1"
ARRAY="--array=50,70,90,110,130,150,170,190,210,230,250,270,290,310,340,370,400,450,500,550,600,650,700,750,800,850,900,950"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
