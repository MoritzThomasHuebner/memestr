#!/usr/bin/env bash


for i in {998..0}
do
    bash production_submit.sh ${i}_pypolychord
done

