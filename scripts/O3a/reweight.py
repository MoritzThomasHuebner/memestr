import pickle
from copy import deepcopy
import numpy as np
from scipy.special import logsumexp
from collections import namedtuple

import bilby

import memestr


Event = namedtuple("Event", ["time_tag", "name", "detectors"])

events = [
    Event(time_tag="1126259462-391", name="GW150914", detectors="H1L1"),
    Event(time_tag="1128678900-4", name="GW151012", detectors="H1L1"),
    Event(time_tag="1135136350-6", name="GW151226", detectors="H1L1"),
    Event(time_tag="1167559936-6", name="GW170104", detectors="H1L1"),
    Event(time_tag="1180922494-5", name="GW170608", detectors="H1L1"),
    Event(time_tag="1185389807-3", name="GW170729", detectors="H1L1V1"),
    Event(time_tag="1186302519-7", name="GW170809", detectors="H1L1V1"),
    Event(time_tag="1186741861-5", name="GW170814", detectors="H1L1V1"),
    Event(time_tag="1187058327-1", name="GW170818", detectors="H1L1V1"),
    Event(time_tag="1187529256-5", name="GW170823", detectors="H1L1"),
    Event(time_tag="1238782700.3", name="GW190408", detectors="H1L1V1"),
    Event(time_tag="1239082262.2", name="GW190412", detectors="H1L1V1"),
    Event(time_tag="1239168612.5", name="GW190413A", detectors="H1L1V1"),
    Event(time_tag="1239198206.7", name="GW190413B", detectors="H1L1V1"),
    Event(time_tag="1239917954.3", name="GW190421A", detectors="H1L1"),
    Event(time_tag="1240164426.1", name="GW190424A", detectors="L1"),
    Event(time_tag="1240215503.0", name="GW190425", detectors="L1V1"),
    Event(time_tag="1240327333.3", name="GW190426A", detectors="H1L1V1"),
    Event(time_tag="1240944862.3", name="GW190503A", detectors="H1L1V1"),
    Event(time_tag="1241719652.4", name="GW190512A", detectors="H1L1V1"),
    Event(time_tag="1241816086.8", name="GW190513A", detectors="H1L1V1"),
    Event(time_tag="1241852074.8", name="GW190514A", detectors="H1L1"),
    Event(time_tag="1242107479.8", name="GW190517A", detectors="H1L1V1"),
    Event(time_tag="1242315362.4", name="GW190519A", detectors="H1L1V1"),
    Event(time_tag="1242442967.4", name="GW190521", detectors="H1L1V1"),
    Event(time_tag="1242459857.5", name="GW190521A", detectors="H1L1"),
    Event(time_tag="1242984073.8", name="GW190527A", detectors="H1L1"),
    Event(time_tag="1243533585.1", name="GW190602A", detectors="H1L1V1"),
    Event(time_tag="1245035079.3", name="GW190620A", detectors="L1V1"),
    Event(time_tag="1245955943.2", name="GW190630A", detectors="L1V1"),
    Event(time_tag="1246048404.6", name="GW190701A", detectors="H1L1V1"),
    Event(time_tag="1246487219.3", name="GW190706A", detectors="H1L1V1"),
    Event(time_tag="1246527224.2", name="GW190707A", detectors="H1L1"),
    Event(time_tag="1246663515.4", name="GW190708A", detectors="L1V1"),
    Event(time_tag="1247608532.9", name="GW190719A", detectors="H1L1"),
    Event(time_tag="1247616534.7", name="GW190720A", detectors="H1L1V1"),
    Event(time_tag="1248242632.0", name="GW190727A", detectors="H1L1V1"),
    Event(time_tag="1248331528.5", name="GW190728A", detectors="H1L1V1"),
    Event(time_tag="1248617394.6", name="GW190731A", detectors="H1L1"),
    Event(time_tag="1248834439.9", name="GW190803A", detectors="H1L1V1"),
    Event(time_tag="1249852257.0", name="GW190814", detectors="L1V1"),
    Event(time_tag="1251009263.8", name="GW190828A", detectors="H1L1V1"),
    Event(time_tag="1251010527.9", name="GW190828B", detectors="H1L1V1"),
    Event(time_tag="1252064527.7", name="GW190909A", detectors="H1L1"),
    Event(time_tag="1252150105.3", name="GW190910A", detectors="L1V1"),
    Event(time_tag="1252627040.7", name="GW190915A", detectors="H1L1V1"),
    Event(time_tag="1253326744.8", name="GW190924A", detectors="H1L1V1"),
    Event(time_tag="1253755327.5", name="GW190929A", detectors="H1L1V1"),
    Event(time_tag="1253885759.2", name="GW190930A", detectors="H1L1")
]

# event_number = int(sys.argv[1])
event_number = 0
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
