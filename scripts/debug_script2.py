import bilby
import memestr

for i in range(2000):
    try:
        bilby.result.read_in_result(str(i) + '_dynesty_production_IMR_non_mem_rec/time_and_phase_shifted_combined_result.json')
    except OSError:
        print(i)