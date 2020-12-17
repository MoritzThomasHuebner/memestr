#!/bin/bash

for mode in 0 1
do
  for run_id in {0..25}
  do
    sbatch submit.sh ${mode} ${run_id}
  done
done