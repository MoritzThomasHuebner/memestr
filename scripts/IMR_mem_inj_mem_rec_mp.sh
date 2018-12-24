#!/usr/bin/env bash
source setup_distances_run_mp.sh ${0}

sbatch ${JOB_NAME} ${OUTPUT} ${TIME} ${NTASKS} ${MEM_PER_CPU} ${EMAIL} <<EOF
#!/usr/bin/env bash
JOB=multiprocessed_run.py
SCRIPT=run_basic_injection_imr_phenom
INJECTION_MODEL=time_domain_IMRPhenomD_waveform_with_memory
RECOVERY_MODEL=time_domain_IMRPhenomD_waveform_with_memory
for i in {1..${NUMBER_OF_TASKS}}
do
srun python \${JOB} ${OUTDIR} \${SCRIPT} \${INJECTION_MODEL} \${RECOVERY_MODEL} ${NUMBER_OF_TASKS} --output=slurm-%j_\$i.out $@
done
EOF