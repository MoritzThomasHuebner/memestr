#!/usr/bin/env bash
JOB=multiprocessed_run.py
OUTDIR="debug_mp"
SCRIPT=run_basic_injection
INJECTION_MODEL=time_domain_IMRPhenomD_waveform_without_memory
RECOVERY_MODEL=time_domain_IMRPhenomD_waveform_without_memory
DISTANCE=800
python3 ${JOB} ${OUTDIR} ${SCRIPT} ${INJECTION_MODEL} ${RECOVERY_MODEL} luminosity_distance=800