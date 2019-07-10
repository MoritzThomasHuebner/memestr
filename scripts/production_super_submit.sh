#!/usr/bin/env bash

#for i in {1850..2000}
#do
#    bash production_IMR_non_mem_rec_submit.sh ${i}_dynesty
#done
#input="n_effs_additional_runs"
#i=0
#while IFS= read -r line
#do
#  if [[ ${line} -gt "0" ]]
#  then
#    for ((j=0; j<=${line}; j++)); do
#      bash production_IMR_non_mem_rec_submit.sh ${i}_dynesty ${j}
#    done
#  fi
#  ((i++))
#done < "$input"

for i in 1863 1870 1876 1896 1903 1907 1912 1916 1936 1937 1938 1958 1971 1982 1996
do
    for j in {100..109}
    do
        bash production_IMR_non_mem_rec_submit.sh ${i}_dynesty_nr_sur ${j}
    done
done

#for i in {1999..1850};
#do
#    bash production_IMR_reweight_submit.sh ${i}_dynesty_nr_sur
#done

