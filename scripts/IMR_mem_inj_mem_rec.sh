#!/usr/bin/env bash

source default_slurm_setup.sh ${0}

sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${CPUS_PER_TASK} ${EMAIL} --export=JOB_NAME=${JOB_NAME}<<'EOF'
#!/usr/bin/env bash
JOB=run_basic_job.py
NAMING_SCHEME=$LABEL
SCRIPT=run_basic_injection_imr_phenom
INJECTION_MODEL=time_domain_IMRPhenomD_waveform_with_memory
RECOVERY_MODEL=time_domain_IMRPhenomD_waveform_with_memory
srun python ${JOB} ${NAMING_SCHEME} ${SCRIPT} ${INJECTION_MODEL} ${RECOVERY_MODEL}
EOF