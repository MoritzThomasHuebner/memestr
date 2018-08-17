#!/usr/bin/env bash

JOB_NAME="--job_name=$1"
OUTPUT="--output=$1.log"
TIME="--time=72:00:00"
NTASKS="--ntasks=1"
MEM_PER_CPU="--mem-per-cpu=16G"
CPUS_PER_TASK="--cpus-per-task=1"
EMAIL="--mail-type=END --mail-user=email@moritz-huebner.de"

sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${CPUS_PER_TASK} ${EMAIL}<<'EOF'
JOB=run_basic_job.py
NAMING_SCHEME=$1
SCRIPT=run_basic_injection_imr_phenom
INJECTION_MODEL=time_domain_IMRPhenomD_waveform_with_memory
RECOVERY_MODEL=time_domain_IMRPhenomD_waveform_with_memory
srun python ${JOB} ${NAMING_SCHEME} ${SCRIPT} ${INJECTION_MODEL} ${RECOVERY_MODEL}
EOF
