for i in {0..50}
do
  for j in {0..4}
  do
    sbatch reweight_submit.sh ${i} ${j}
  done
done