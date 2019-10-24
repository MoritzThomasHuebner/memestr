#!/usr/bin/env bash

bash production_IMR_non_mem_rec.sh              outdir_base=${1} filename_base=${1} sub_run_id=${2} script=run_production_injection recovery_model=frequency_domain_IMRPhenomD_waveform_with_memory npoints=1000 alpha=0.1 distance_marginalization=True time_marginalization=False phase_marginalization=False resume=False clean=True sampler=dynesty duration=16 random_seed=42 sampling_frequency=2048 label=IMR_mem_inj_non_mem_rec
#recovery_model=time_domain_IMRPhenomD_waveform_without_memory