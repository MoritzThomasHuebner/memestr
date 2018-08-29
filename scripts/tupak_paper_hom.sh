#!/usr/bin/env bash
bash ../install.sh
bash NRSur_non_mem_inj_non_mem_rec.sh luminosity_distance=50 l_max=2 mass_ratio=1.2414 total_mass=65 npoints=5000 iota=0.4 psi=2.659 phase=1.3 geocent_time=1126259642.413 ra=1.375 dec=-1.2108
sleep 10
bash NRSur_non_mem_inj_non_mem_rec.sh luminosity_distance=50 l_max=4 mass_ratio=1.2414 total_mass=65 npoints=5000 iota=0.4 psi=2.659 phase=1.3 geocent_time=1126259642.413 ra=1.375 dec=-1.2108