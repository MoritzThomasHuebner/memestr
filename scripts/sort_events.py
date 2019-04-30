import bilby as bb
import numpy as np
import pandas as pd
from shutil import copyfile


ids = []
snrs = []

for i in range(1000):
    ifos = bb.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(i) + '_H1L1V1.h5')
    best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
    network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
    ids.append(i)
    snrs.append(network_snr)

data = pd.DataFrame({'ids': ids, 'snr': snrs})
data = data.sort_values(by='snr')
print(data)

for new_id, old_id in enumerate(data.id):
    src = 'parameter_sets/' + str(old_id) + '_H1V1L1.h5'
    dst = 'parameter_sets/' + str(new_id) + '_H1V1L1_new.h5'
    copyfile(src, dst)
    src = 'parameter_sets/' + str(old_id)
    dst = 'parameter_sets/' + str(new_id)
    copyfile(src, dst)

