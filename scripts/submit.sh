#!/usr/bin/env bash
bash ../install.sh
sbatch IMR_mem_inj_mem_rec.sh IMR_mem_inj_mem_rec
sbatch IMR_mem_inj_non_mem_rec.sh IMR_mem_inj_non_mem_rec
sbatch NRSur_mem_inj_mem_rec.sh NRSur_mem_inj_mem_rec
sbatch NRSur_mem_inj_non_mem_rec.sh NRSur_mem_inj_non_mem_rec