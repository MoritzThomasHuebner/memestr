#!/usr/bin/env bash


for i in {1999..0}
do
    bash production_submit.sh ${i}_dynesty
done

