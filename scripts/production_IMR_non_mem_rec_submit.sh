#!/usr/bin/env bash

bash production_IMR_non_mem_rec.sh              --outdir ${1} --sub_run_id ${2} --routine run_production_injection --recovery_model fd_imrd_with_memory --npoints 1000 distance_marginalization=True --resume False --clean True label=IMR_mem_inj_non_mem_rec
