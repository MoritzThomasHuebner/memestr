import numpy as np
import json

network_snrs = []
ln_bfs = []

for i in range(2000):
    try:
        with open(f'injections/{i}_memory/IMR_mem_inj_non_mem_rec_result.json') as f:
            res = json.load(f)
    except Exception as e:
        print(e)
    with open(f'injections/{i}_memory/pp_result.json') as f:
        pp_res = json.load(f)

    snrs_squared = [res['meta_data']['likelihood']['interferometers'][ifo]['optimal_SNR']**2 for ifo in ['H1', 'L1', 'V1']]
    network_snrs.append(np.sqrt(np.sum(snrs_squared)))
    ln_bfs.append(pp_res['memory_log_bf'])
    print(i)
print()
ln_bfs = np.array(ln_bfs)
network_snrs = np.array(network_snrs)

low_snr_indices = np.where(network_snrs < 15)
high_snr_indices = np.where(network_snrs > 15)
print(len(low_snr_indices))
print()
print(np.mean(ln_bfs[low_snr_indices]))
print(np.std(ln_bfs[low_snr_indices]))
print()
print(np.mean(ln_bfs[high_snr_indices]))
print(np.std(ln_bfs[high_snr_indices]))
