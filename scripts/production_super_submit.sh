#!/usr/bin/env bash

input="n_effs_additional_runs"
i=0
while IFS= read -r line
do
  if [[ ${line} -gt "0" ]]
  then
    for ((j=0; j<=${line}; j++)); do
      echo "${i}_dynesty ${j}"
    done
  fi
  ((i++))
done < "$input"


#bash production_submit.sh