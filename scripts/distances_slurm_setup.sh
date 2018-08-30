#!/usr/bin/env bash

LABEL="${1::-3}"
PYTHON_COMMAND="import memestr; print(memestr.submit.submitter. d_unallocated_name(name=\"$LABEL\"))"
OUTDIR=`python -c "${PYTHON_COMMAND}"`
JOB_NAME="--job-name=$LABEL"
OUTPUT="--output=$OUTDIR.log"
TIME="--time=72:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=16G"
CPUS_PER_TASK="--cpus-per-task=1"
ARRAY="--array=10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,29,30,32,34,36,38,40,42,44,47,49,52,55,58,61,65,68,72,76,80,85,89,94,99,105,111,117,123,130,137,145,153,161,170,179,189,200,211,222,235,248,261,276,291,307,324,341,360,380,401,423,446,471,497,524,553,584,616,650,685,723,763,805,849,896,945,997,1052,1110,1171,1235,1303,1375,1450,1530,1614,1703,1796,1895,2000"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
