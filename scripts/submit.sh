#!/usr/bin/env bash
bash IMR_mem_inj_mem_rec.sh           npoints=5000 zero_noise=False inc=0.5 sampler=dynesty label=IMR_mem_inj_mem_rec
bash IMR_mem_inj_non_mem_rec.sh       npoints=5000 zero_noise=False inc=0.5 sampler=dynesty label=IMR_mem_inj_non_mem_rec
bash IMR_non_mem_inj_non_mem_rec.sh   npoints=5000 zero_noise=False inc=0.5 sampler=dynesty label=IMR_non_mem_inj_non_mem_rec
bash IMR_non_mem_inj_mem_rec.sh       npoints=5000 zero_noise=False inc=0.5 sampler=dynesty label=IMR_non_mem_inj_mem_rec
#bash NRSur_mem_inj_mem_rec.sh          npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=NRSur_mem_inj_mem_rec
#bash NRSur_non_mem_inj_mem_rec.sh      npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=NRSur_non_mem_inj_mem_rec
#bash NRSur_mem_inj_non_mem_rec.sh      npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=NRSur_mem_inj_non_mem_rec
#bash NRSur_non_mem_inj_non_mem_rec.sh  npoints=1000 zero_noise=False l_max=4 sampler=dynesty label=NRSur_non_mem_inj_non_mem_rec