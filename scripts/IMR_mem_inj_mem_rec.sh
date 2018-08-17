#!/usr/bin/env bash
#
#SBATCH --job-name=IMR_mem_inj_mem_rec
#SBATCH --output=IMR_mem_inj_mem_rec.log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END --mail-user=email@moritz-huebner.de

srun python run_basic_job.py IMR_mem_inj_mem_rec run_basic_injection_imr_phenom time_domain_IMRPhenomD_waveform_with_memory time_domain_IMRPhenomD_waveform_with_memory
