import bilby
import os

for i in range(2000):
    if os.path.isfile(str(i) + '_dynesty_production_IMR_non_mem_rec/time_and_phase_shifted_combined_result.json'):
        continue
    else:
        print(i)