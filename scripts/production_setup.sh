#!/usr/bin/env bash

LABEL="production_IMR_non_mem_rec"
OUTDIR=${2//outdir_base=/}_${LABEL}
mkdir -p ${OUTDIR}
JOB_NAME="--job-name=$LABEL"
OUTPUT=""
TIME="--time=24:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=1G"
CPUS_PER_TASK="--cpus-per-task=1"
#EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"
ARRAY=""
