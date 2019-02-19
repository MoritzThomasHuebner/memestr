#!/usr/bin/env bash

#Distance vs evidence
#bash IMR_mem_inj_mem_rec.sh                 npoints=5000 alpha=0.1 dlogz=0.1 zero_noise=False sampler=dynesty luminosity_distance=$1 ra=0.684 dec=0.672 psi=4.28 distance_marginalization=True label=IMR_mem_inj_mem_rec
#bash IMR_mem_inj_non_mem_rec.sh             npoints=5000 alpha=0.1 dlogz=0.1 zero_noise=False sampler=dynesty luminosity_distance=$1 ra=0.684 dec=0.672 psi=4.28 distance_marginalization=True label=IMR_mem_inj_non_mem_rec

#Population run
bash IMR_mem_inj_mem_rec.sh                 npoints=2000 maxmcmc=2000 alpha=0.1 zero_noise=False distance_marginalization=True sampler=cpnest nthreads=4 duration=16 label=IMR_mem_inj_mem_rec
bash IMR_mem_inj_non_mem_rec.sh             npoints=2000 maxmcmc=2000 alpha=0.1 zero_noise=False distance_marginalization=True sampler=cpnest nthreads=4 duration=16 label=IMR_mem_inj_non_mem_rec
