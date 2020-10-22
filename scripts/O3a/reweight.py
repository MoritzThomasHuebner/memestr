import pickle
from copy import deepcopy
import numpy as np
from scipy.special import logsumexp

import bilby

import memestr

# event_number = int(sys.argv[1])
event_number = 0
time_tags = ["1126259462-391", "1128678900-4", "1135136350-6", "1167559936-6", "1180922494-5", "1185389807-3",
             "1186302519-7", "1186741861-5", "1187058327-1", "1187529256-5", "1239082262-222168"]
events = ["GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
          "GW170809", "GW170814", "GW170818", "GW170823", "GW190412"]
time_tag = time_tags[event_number]
event = events[event_number]

try:
    result = bilby.core.result.read_in_result('{}_precessing_22/result/run_data0_{}_analysis_H1L1_dynesty_merge_result.json'.format(event, time_tag))
except Exception:
    result = bilby.core.result.read_in_result(
        '{}_precessing_22/result/run_data0_{}_analysis_H1L1V1_dynesty_merge_result.json'.format(event, time_tag))
print(len(result.posterior))

# result.plot_corner()

with open('{}_precessing_22/data/run_data0_{}_generation_data_dump.pickle'.format(event, time_tag), "rb") as f:
    data_dump = pickle.load(f)
ifos = data_dump.interferometers


bilby.core.utils.logger.disabled = True
wg_xphm_hom = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.core.waveforms.fd_imrxp) #, waveform_arguments=dict(alpha=0.1))
wg_xphm_22 = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.core.waveforms.fd_imrxp_22)
wg_xphm_lal = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM"))
bilby.core.utils.logger.disabled = False



# likelihood_22 = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=wg_xphm_22)
# likelihood_hom = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=wg_xphm_hom)
# likelihood_22.parameters = sample
# likelihood_hom.parameters = sample
# print(likelihood_22.log_likelihood_ratio())
# print(sample['log_likelihood'])
# print(likelihood_hom.log_likelihood_ratio())
# print()

sample = result.posterior.iloc[-1]

wg_xphm_22.parameters = dict(sample)
wg_xphm_hom.parameters = dict(sample)
wg_xphm_lal.parameters = dict(sample)

import matplotlib.pyplot as plt
plt.plot(wg_xphm_22.time_array, wg_xphm_22.time_domain_strain()['plus'])
# plt.plot(wg_xphm_hom.time_array, wg_xphm_hom.time_domain_strain()['plus'])
plt.plot(wg_xphm_lal.time_array, wg_xphm_lal.time_domain_strain()['plus'])
plt.savefig('test_td_plus.pdf')
plt.clf()

plt.plot(wg_xphm_22.time_array, wg_xphm_22.time_domain_strain()['cross'])
# plt.plot(wg_xphm_hom.time_array, wg_xphm_hom.time_domain_strain()['cross'])
plt.plot(wg_xphm_lal.time_array, wg_xphm_lal.time_domain_strain()['cross'])
plt.savefig('test_td_cross.pdf')
plt.clf()

plt.loglog(wg_xphm_22.frequency_array, np.abs(wg_xphm_22.frequency_domain_strain()['plus']))
# plt.loglog(wg_xphm_hom.frequency_array, np.abs(wg_xphm_hom.frequency_domain_strain()['plus']))
plt.plot(wg_xphm_lal.frequency_array, np.abs(wg_xphm_lal.frequency_domain_strain()['plus']))
plt.savefig('test_fd_plus.pdf')
plt.clf()

plt.loglog(wg_xphm_22.frequency_array, np.abs(wg_xphm_22.frequency_domain_strain()['cross']))
# plt.loglog(wg_xphm_hom.frequency_array, np.abs(wg_xphm_hom.frequency_domain_strain()['cross']))
plt.plot(wg_xphm_lal.frequency_array, np.abs(wg_xphm_lal.frequency_domain_strain()['cross']))
plt.savefig('test_fd_cross.pdf')
plt.clf()

# reweighted_log_bf, log_weights = memestr.core.postprocessing.reweigh_by_likelihood(
#     new_likelihood=likelihood_hom, new_result=result, reference_likelihood=likelihood_22, reference_result=None)
#
# np.savetxt("{}_log_weights".format(event), log_weights)
#
#
# reweighted_log_bf = logsumexp(log_weights) - np.log(len(log_weights))
# print(reweighted_log_bf)





# log_bfs = []
#
# for event in events:
#     log_weights = np.loadtxt("{}_log_weights".format(event))
#     reweighted_log_bf = logsumexp(log_weights) - np.log(len(log_weights))
#     print(reweighted_log_bf)
#     log_bfs.append(reweighted_log_bf)
#
# gwtm_1_original = np.loadtxt("GWTM-1.txt")
#
# import matplotlib
# matplotlib.rcParams.update({'font.size': 13})
# import matplotlib.pyplot as plt
#
# plt.scatter(np.arange(0, 11), log_bfs, label="New (PhenomXHM)")
# plt.scatter(np.arange(0, 10), gwtm_1_original, label="Huebner et al. (NRHybSur)")
# plt.xticks(ticks=np.arange(0, 11), labels=events, rotation=75)
# plt.ylabel("ln BF")
# plt.legend()
# plt.tight_layout()
# plt.savefig("gwtm-1.png")
# plt.clf()
