#!/usr/bin/env bash
bash IMR_mem_inj_mem_rec.sh           conversion_function=convert_to_lal_binary_black_hole_parameters npoints=5000 zero_noise=True label=IMR_mem_inj_mem_rec
bash IMR_mem_inj_non_mem_rec.sh       conversion_function=convert_to_lal_binary_black_hole_parameters npoints=5000 zero_noise=True label=IMR_mem_inj_non_mem_rec
bash IMR_non_mem_inj_non_mem_rec.sh   conversion_function=convert_to_lal_binary_black_hole_parameters npoints=5000 zero_noise=True label=IMR_non_mem_inj_non_mem_rec
bash IMR_non_mem_inj_mem_rec.sh       conversion_function=convert_to_lal_binary_black_hole_parameters npoints=5000 zero_noise=True label=IMR_non_mem_inj_mem_rec
bash IMR_pure_mem_inj_pure_mem_rec.sh conversion_function=convert_to_lal_binary_black_hole_parameters npoints=5000 zero_noise=True label=IMR_pure_mem_inj_pure_mem_rec
#bash NRSur_mem_inj_mem_rec.sh
#bash NRSur_mem_inj_non_mem_rec.sh