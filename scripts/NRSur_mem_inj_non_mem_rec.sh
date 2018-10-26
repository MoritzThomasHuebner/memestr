#!/usr/bin/env bash

#source setup_default.sh ${0}
source setup_distances_run.sh ${0}


sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${CPUS_PER_TASK} ${EMAIL} ${ARRAY}<<EOF
#!/usr/bin/env bash
JOB=run_basic_job.py
SCRIPT=run_basic_injection_nrsur
INJECTION_MODEL=time_domain_nr_sur_waveform_with_memory
RECOVERY_MODEL=time_domain_nr_sur_waveform_without_memory
FILENAME="./parameter_sets/\${SLURM_ARRAY_TASK_ID}"
#while IFS= read -r var
#do
#  PARAMS="\$var"
#done < "\$FILENAME"
PARAMS=""
srun python \${JOB} ${OUTDIR}/\${SLURM_ARRAY_TASK_ID} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL}  \${PARAMS} luminosity_distance=\${SLURM_ARRAY_TASK_ID} $@
EOF