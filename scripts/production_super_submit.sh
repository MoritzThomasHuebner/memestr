#!/usr/bin/env bash

for i in {100..132}
do
    sbatch production_submit.sh ${i}
done