#!/usr/bin/env bash

source production_setup.sh ${0} ${1}

sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${CPUS_PER_TASK} ${EMAIL} ${ARRAY}<<EOF
#!/usr/bin/env bash
JOB=run_basic_job.py
SCRIPT=run_production_injection_imr_phenom
INJECTION_MODEL=time_domain_nr_hyb_sur_waveform_with_memory_wrapped
RECOVERY_MODEL=frequency_domain_IMRPhenomD_waveform_without_memory

srun python \${JOB} ${OUTDIR}/\${SLURM_ARRAY_TASK_ID} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL} $@
EOF