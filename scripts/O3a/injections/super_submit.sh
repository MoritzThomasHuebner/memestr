#!/bin/bash

for mode in 0 1
do
  for run_id in {0..99}
  do
    sbatch submit.sh ${1} ${2}
  done
done