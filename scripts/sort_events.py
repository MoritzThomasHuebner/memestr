import bilby as bb
import numpy as np
import pandas as pd

ids = []
snrs = []

for i in range(100):
    ifos = bb.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(i) + '_H1L1V1.h5')
    best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
    network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
    ids.append(i)
    snrs.append(network_snr)

data = pd.DataFrame({'id': ids, 'snr': snrs})
data = data.sort_values(by='snr')
print(data)
