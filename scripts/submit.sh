#!/usr/bin/env bash
#bash IMR_mem_inj_mem_rec.sh           npoints=10000 zero_noise=False inc=0.5 sampler=dynesty label=I_m_m
#bash IMR_mem_inj_non_mem_rec.sh       npoints=10000 zero_noise=False inc=0.5 sampler=dynesty label=I_m_n
#bash IMR_non_mem_inj_non_mem_rec.sh   npoints=10000 zero_noise=False inc=0.5 sampler=dynesty label=I_n_n
#bash IMR_non_mem_inj_mem_rec.sh       npoints=10000 zero_noise=False inc=0.5 sampler=dynesty label=I_n_m
#bash IMR_pure_mem_inj_pure_mem_rec.sh time_marginalization=True npoints=5000 zero_noise=False random_injection_parameters=True label=IMR_pure_mem_inj_pure_mem_rec
bash NRSur_mem_inj_mem_rec.sh          npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=N_m_m
bash NRSur_non_mem_inj_mem_rec.sh      npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=N_n_n
bash NRSur_mem_inj_non_mem_rec.sh      npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=N_m_n
bash NRSur_non_mem_inj_non_mem_rec.sh  npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=N_n_n