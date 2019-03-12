#!/usr/bin/env bash

bash IMR_mem_inj_mem_rec.sh                 npoints=2000 maxmcmc=1000 alpha=0.1 zero_noise=True distance_marginalization=True time_marginalization=True luminosity_distance=$1 sampler=cpnest nthreads=4 duration=16 random_seed=42 sampling_frequency=2048 label=IMR_mem_inj_mem_rec
bash IMR_mem_inj_non_mem_rec.sh             npoints=2000 maxmcmc=1000 alpha=0.1 zero_noise=True distance_marginalization=True time_marginalization=True luminosity_distance=$1 sampler=cpnest nthreads=4 duration=16 random_seed=42 sampling_frequency=2048 label=IMR_mem_inj_non_mem_rec


#bash IMR_non_mem_inj_mem_rec.sh                 npoints=2000 maxmcmc=1000 alpha=0.1 zero_noise=True distance_marginalization=True time_marginalization=True luminosity_distance=$1 sampler=cpnest nthreads=4 duration=16 random_seed=42 sampling_frequency=2048 label=IMR_non_mem_inj_mem_rec
#bash IMR_non_mem_inj_non_mem_rec.sh             npoints=2000 maxmcmc=1000 alpha=0.1 zero_noise=True distance_marginalization=True time_marginalization=True luminosity_distance=$1 sampler=cpnest nthreads=4 duration=16 random_seed=42 sampling_frequency=2048 label=IMR_non_mem_inj_non_mem_rec
