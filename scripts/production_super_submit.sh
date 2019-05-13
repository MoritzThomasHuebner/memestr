#!/usr/bin/env bash


for i in {999..0}
do
    bash production_submit.sh ${i}_pypolychord
done

