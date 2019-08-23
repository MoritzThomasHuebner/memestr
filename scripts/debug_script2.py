import bilby
import numpy as np

network_snrs = []
for run_id in range(48):
    run_id = str(run_id).zfill(3)
    res = bilby.result.read_in_result('SNR_VS_LOGBF_DATA/{}_IMR_mem_inj_non_mem_rec/0/IMR_mem_inj_non_mem_rec_result.json'.format(run_id))
    network_snr = np.sqrt(np.sum([res.meta_data['likelihood']['interferometers'][ifo]['optimal_SNR']**2 for ifo in ['H1', 'V1', 'L1']]))
    print(network_snr)
    network_snrs.append(network_snr)


np.savetxt('SNR_VS_LOGBF_DATA/snrs.txt', network_snrs)