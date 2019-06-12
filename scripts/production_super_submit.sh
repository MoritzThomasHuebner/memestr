#!/usr/bin/env bash

input="n_effs_additional_runs"
i=0
while IFS= read -r line
do
  if [$line -gt "0"]
  then
    echo "$line"
    for j in {0..$line}
    do
#      bash production_submit.sh ${i}_dynesty j
       echo $j
    done
  fi
  echo $i
  ((i++))

done <<< $(command)

for i in {1999..0}
do

done

