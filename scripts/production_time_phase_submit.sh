#!/usr/bin/env bash

bash production_run.sh               --outdir=${1} --sub_run_id ${2} --routine run_time_phase_optimization --recovery_model fd_nr_sur --distance_marginalization True --resume True --sampler dynesty --label IMR_mem_inj_non_mem_rec