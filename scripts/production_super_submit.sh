#!/usr/bin/env bash


for i in {1400..1500}
do
    bash production_submit.sh ${i}_pypolychord
done

