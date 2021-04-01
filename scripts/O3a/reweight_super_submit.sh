for i in {0..50}
do
#  sbatch reweight_submit.sh ${i} 0
  sbatch reweight_submit.sh ${i} 1 20
done