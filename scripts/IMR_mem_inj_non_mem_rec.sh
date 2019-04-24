#!/usr/bin/env bash
#source setup_distances_run.sh ${0}
echo ${0}
echo ${1}
source setup_default.sh ${0} ${1}
#source setup_random_injections.sh ${0} ${1}

sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${CPUS_PER_TASK} ${EMAIL} ${ARRAY}<<EOF
#!/usr/bin/env bash
JOB=run_basic_job.py
SCRIPT=run_basic_injection_imr_phenom
INJECTION_MODEL=time_domain_nr_hyb_sur_waveform_with_memory_wrapped
RECOVERY_MODEL=frequency_domain_IMRPhenomD_waveform_without_memory

#srun python \${JOB} ${OUTDIR} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL} $@

#Random injections
#FILENAME="./parameter_sets/\${SLURM_ARRAY_TASK_ID}"
#PARAMS=\$(cat \$FILENAME)
#srun python \${JOB} ${OUTDIR}/\${SLURM_ARRAY_TASK_ID} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL} \${PARAMS} random_seed=\${SLURM_ARRAY_TASK_ID} $@

#FILENAME="./parameter_sets/0"
#PARAMS=\$(cat \$FILENAME)
#srun python \${JOB} ${OUTDIR}/\${SLURM_ARRAY_TASK_ID} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL} \${PARAMS} $@

#Distance vs evidence
srun python \${JOB} ${OUTDIR}/\${SLURM_ARRAY_TASK_ID} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL} $@
EOF