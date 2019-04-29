#!/usr/bin/env bash
# sbatch evidence_reweighing_submit.sh 000
for i in {100..132}
do
    sbatch evidence_reweighing_submit.sh ${i}
done
