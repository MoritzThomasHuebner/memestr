import bilby as bb
import numpy as np
import pandas as pd
import os
from shutil import copyfile
import matplotlib.pyplot as plt

ids = []
snrs = []
new_ids = []

for i in range(0, 2000):
    ifos = bb.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(i) + '_H1L1V1.h5')
    best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
    network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
    ids.append(i)
    snrs.append(network_snr)
    new_ids.append(i)

data = pd.DataFrame({'ids': ids, 'snr': snrs})
data = data.sort_values(by='snr')
np.savetxt('snrs.txt', data.values)

plt.hist(snrs, bins=32)
plt.savefig('snrs')
plt.clf()


for new_id, old_id in zip(new_ids, data.ids):
    src = 'parameter_sets/' + str(old_id) + '_H1L1V1.h5'
    dst = 'parameter_sets/' + str(new_id) + '_H1L1V1_new.h5'
    copyfile(src, dst)
    src = 'parameter_sets/' + str(old_id)
    dst = 'parameter_sets/' + str(new_id) + '_new'
    copyfile(src, dst)

for old_id in range(len(data.ids)):
    os.remove('parameter_sets/' + str(old_id))
    os.remove('parameter_sets/' + str(old_id) + '_H1L1V1.h5')
    os.rename('parameter_sets/' + str(old_id) + '_new', 'parameter_sets/' + str(old_id))
    os.rename('parameter_sets/' + str(old_id) + '_H1L1V1_new.h5', 'parameter_sets/' + str(old_id) + '_H1L1V1.h5')
