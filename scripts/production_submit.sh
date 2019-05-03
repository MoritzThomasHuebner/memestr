#!/usr/bin/env bash

bash production_IMR_non_mem_rec.sh              outdir_base=${1} filename_base=${1} npoints=500 alpha=0.1 distance_marginalization=True time_marginalization=False phase_marginalization=True resume=True sampler=pypolychord duration=16 random_seed=42 sampling_frequency=2048 label=IMR_mem_inj_non_mem_rec
