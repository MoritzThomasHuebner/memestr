#!/usr/bin/env bash
#
#SBATCH --job-name=NRSur_mem_inj_non_mem_rec
#SBATCH --output=NRSur_mem_inj_non_mem_rec.log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de

JOB=run_basic_job.py
NAMING_SCHEME=NRSur_mem_inj_mem_rec
SCRIPT=run_basic_injection
INJECTION_MODEL=time_domain_nr_sur_waveform_with_memory
RECOVERY_MODEL=time_domain_nr_sur_waveform_without_memory
srun python ${JOB} ${NAMING_SCHEME} ${SCRIPT} ${INJECTION_MODEL} ${RECOVERY_MODEL}