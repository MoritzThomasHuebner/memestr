#!/usr/bin/env bash
bash IMR_mem_inj_mem_rec.sh           distance_marginalization=True time_marginalization=True npoints=5000 zero_noise=True label=IMR_mem_inj_mem_rec
bash IMR_mem_inj_non_mem_rec.sh       distance_marginalization=True time_marginalization=True npoints=5000 zero_noise=True label=IMR_mem_inj_non_mem_rec
bash IMR_non_mem_inj_non_mem_rec.sh   distance_marginalization=True time_marginalization=True npoints=5000 zero_noise=True label=IMR_non_mem_inj_non_mem_rec
bash IMR_non_mem_inj_mem_rec.sh       distance_marginalization=True time_marginalization=True npoints=5000 zero_noise=True label=IMR_non_mem_inj_mem_rec
bash IMR_pure_mem_inj_pure_mem_rec.sh distance_marginalization=True time_marginalization=True npoints=5000 zero_noise=True label=IMR_pure_mem_inj_pure_mem_rec
#bash NRSur_mem_inj_mem_rec.sh
#bash NRSur_mem_inj_non_mem_rec.sh