#!/usr/bin/env bash

for i in {20000..20029}
do
    for j in {0..7}
    do
        OUTDIR="injections/${i}_memory"
        mkdir -p ${OUTDIR}
        bash production_IMR_reweight_submit.sh ${OUTDIR} ${j}
    done
done
