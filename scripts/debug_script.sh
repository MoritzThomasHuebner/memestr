#!/usr/bin/env bash
JOB=run_basic_job.py
OUTDIR="test"
SCRIPT=run_basic_injection_imr_phenom
INJECTION_MODEL=time_domain_IMRPhenomD_waveform_with_memory
RECOVERY_MODEL=time_domain_IMRPhenomD_waveform_with_memory
python ${JOB} ${OUTDIR} ${SCRIPT} ${INJECTION_MODEL} ${RECOVERY_MODEL}