#!/usr/bin/env bash

for i in {100..132}
do
    sbatch submit.sh ${i}_polychord
done