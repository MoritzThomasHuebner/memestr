#!/usr/bin/env bash
bash ../install.sh
sbatch IMR_mem_inj_mem_rec.sh
sbatch IMR_mem_inj_non_mem_rec.sh
sbatch NRSur_mem_inj_mem_rec.sh
sbatch NRSur_mem_inj_non_mem_rec.sh