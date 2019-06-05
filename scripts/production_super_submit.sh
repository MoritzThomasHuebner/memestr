#!/usr/bin/env bash


for i in {1000..1999}
do
    bash production_submit.sh ${i}_pypolychord
done

