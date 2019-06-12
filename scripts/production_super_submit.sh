#!/usr/bin/env bash

input="n_effs_additional_runs"
i=0
while IFS= read -r line
do
  if [[ ${line} -gt "0" ]]
  then
    echo "$line"
    for j in {0..$line}
    do
       echo $j
    done
  fi
  echo $i
  ((i++))

done <<< "$input"


