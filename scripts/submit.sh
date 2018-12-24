#!/usr/bin/env bash
#bash IMR_mem_inj_mem_rec.sh                 npoints=5000 alpha=0.1 dlogz=0.02 zero_noise=True sampler=dynesty luminosity_distance=$1 label=IMR_mem_inj_mem_rec
#bash IMR_mem_inj_non_mem_rec.sh             npoints=5000 alpha=0.1 dlogz=0.02 zero_noise=True sampler=dynesty luminosity_distance=$1 label=IMR_mem_inj_non_mem_rec
bash IMR_mem_inj_mem_rec_mp.sh                 npoints=5000 alpha=0.1 dlogz=10 zero_noise=True sampler=dynesty luminosity_distance=$1 label=IMR_mem_inj_mem_rec_mp
bash IMR_mem_inj_non_mem_rec_mp.sh             npoints=5000 alpha=0.1 dlogz=10 zero_noise=True sampler=dynesty luminosity_distance=$1 label=IMR_mem_inj_non_mem_rec_mp
#bash IMR_non_mem_inj_non_mem_rec.sh         npoints=5000 walks=200 zero_noise=True sampler=dynesty label=IMR_non_mem_inj_non_mem_rec
#bash IMR_non_mem_inj_mem_rec.sh             npoints=5000 walks=200 zero_noise=True sampler=dynesty label=IMR_non_mem_inj_mem_rec
#bash IMR_pure_mem_inj_pure_mem_rec.sh       npoints=4000 zero_noise=False sampler=dynesty time_marginalization=True label=IMR_pure_mem_inj_pure_mem_rec
#bash NRSur_mem_inj_mem_rec.sh                     npoints=1000 psi=2.356 phase=0.785 zero_noise=False l_max=4 sampler=dynesty label=NRSur_mem_inj_mem_rec
#bash NRSur_mem_inj_mem_rec_base_modes.sh          npoints=1000 psi=2.356 phase=0.785 zero_noise=False l_max=2 sampler=dynesty label=NRSur_mem_inj_mem_rec
#bash NRSur_non_mem_inj_non_mem_rec.sh             npoints=1000 psi=2.356 phase=0.785 zero_noise=False l_max=4 sampler=dynesty label=NRSur_non_mem_inj_non_mem_rec
#bash NRSur_non_mem_inj_non_mem_rec_base_modes.sh  npoints=1000 psi=2.356 phase=0.785 zero_noise=False l_max=2 sampler=dynesty label=NRSur_non_mem_inj_non_mem_rec
#bash NRSur_non_mem_inj_mem_rec.sh      npoints=1000 psi=2.356 phase=0.785 zero_noise=False l_max=4 sampler=dynesty label=NRSur_non_mem_inj_mem_rec
#bash NRSur_mem_inj_non_mem_rec.sh      npoints=1000 psi=2.356 phase=0.785 zero_noise=False l_max=4 sampler=dynesty label=NRSur_mem_inj_non_mem_rec
