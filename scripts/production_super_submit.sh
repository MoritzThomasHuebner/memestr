#!/usr/bin/env bash


for i in {0..999}
do
    bash production_submit.sh ${i}_pypolychord
done

