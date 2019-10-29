#!/usr/bin/env bash

for i in {20000..20000}
do
    for j in {100..131}
    do
        OUTDIR="injections/${i}_memory"
        mkdir -p ${OUTDIR}
        bash production_reweight_submit.sh ${OUTDIR} ${j}
    done
done
