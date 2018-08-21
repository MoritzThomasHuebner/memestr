#!/usr/bin/env bash

source default_slurm_setup.sh ${0}

sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${CPUS_PER_TASK} ${EMAIL}<<EOF
#!/usr/bin/env bash
JOB=run_basic_job.py
NAMING_SCHEME=${LABEL}
SCRIPT=run_basic_injection_nrsur
INJECTION_MODEL=time_domain_nr_sur_waveform_with_memory
RECOVERY_MODEL=time_domain_nr_sur_waveform_without_memory
srun python \${JOB} \${NAMING_SCHEME} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL}
EOF