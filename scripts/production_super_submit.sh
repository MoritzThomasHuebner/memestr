#!/usr/bin/env bash

for i in {20000..20029}
do
    for j in {10..17}
    do
        bash production_IMR_reweight_submit.sh ${i}_dynesty ${j}
    done
done
