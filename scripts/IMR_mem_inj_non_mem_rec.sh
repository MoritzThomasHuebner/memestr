#!/usr/bin/env bash

#source distances_slurm_setup.sh ${0}
#source default_slurm_setup.sh ${0}
source setup_random_injections.sh ${0}

sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${CPUS_PER_TASK} ${EMAIL} ${ARRAY}<<EOF
#!/usr/bin/env bash
JOB=run_basic_job.py
SCRIPT=run_basic_injection_imr_phenom
INJECTION_MODEL=time_domain_IMRPhenomD_waveform_with_memory
RECOVERY_MODEL=time_domain_IMRPhenomD_waveform_without_memory
bash get_injection_bash_strings.sh \${SLURM_ARRAY_TASK_ID}
srun python \${JOB} ${OUTDIR}/\${SLURM_ARRAY_TASK_ID} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL} \${PARAMS} $@
EOF