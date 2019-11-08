#!/usr/bin/env bash

LABEL="memory"
JOB_NAME="--job-name=$LABEL"
TIME="--time=72:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=1G"
CPUS_PER_TASK="--cpus-per-task=1"
#EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
