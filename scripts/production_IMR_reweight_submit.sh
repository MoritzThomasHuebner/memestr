#!/usr/bin/env bash

bash production_IMR_non_mem_rec.sh               --outdir ${1} --sub_run_id ${2} --routine run_reweighting --reweight_model fd_imrd_with_memory --recovery_model fd_imrd --distance_marginalization True --label IMR_mem_inj_non_mem_rec
