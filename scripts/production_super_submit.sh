#!/usr/bin/env bash


for i in {999..0}
do
    sbatch production_submit.sh ${i}_pypolychord
done

