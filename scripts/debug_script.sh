#!/usr/bin/env bash
JOB=run_basic_job.py
OUTDIR="NRSur_HOM"
SCRIPT=run_basic_job
INJECTION_MODEL=time_domain_nr_sur_waveform_with_memory
RECOVERY_MODEL=time_domain_nr_sur_waveform_with_memory
DISTANCE=200
python ${JOB} ${OUTDIR} ${SCRIPT} ${INJECTION_MODEL} ${RECOVERY_MODEL} luminosity_distance=50 l_max=4