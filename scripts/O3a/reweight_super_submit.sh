for i in {40..78}
do
  sbatch reweight_submit.sh ${i} 0 0
  sbatch reweight_submit.sh ${i} 1 0
done