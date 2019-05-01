#!/usr/bin/env bash

#for i in {970..999}
for i in {0..15}
do
    sbatch production_submit.sh ${i}_dynesty
done