import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

import bilby

import memestr

# event_number = int(sys.argv[1])
time_tags = ["1126259462-391", "1128678900-4", "1135136350-6", "1167559936-6", "1180922494-5", "1185389807-3",
             "1186302519-7", "1186741861-5", "1187058327-1", "1187529256-5", "1239082262-222168"]
events = ["GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
          "GW170809", "GW170814", "GW170818", "GW170823", "GW190412"]
# time_tag = time_tags[event_number]
# event = events[event_number]

# try:
#     result = bilby.core.result.read_in_result('results/{}/result/run_data0_{}_analysis_H1L1_dynesty_merge_result.json'.format(event, time_tag))
# except Exception:
#     result = bilby.core.result.read_in_result(
#         'results/{}/result/run_data0_{}_analysis_H1L1V1_dynesty_merge_result.json'.format(event, time_tag))
# print(len(result.posterior))
# from matplotlib import rc
# rc("text", usetex=False)
# result.plot_corner(outdir='results')
#
# with open('results/{}/data/run_data0_{}_generation_data_dump.pickle'.format(event, time_tag), "rb") as f:
#     data_dump = pickle.load(f)
# ifos = data_dump.interferometers
#
# posterior = result.posterior
#
# parameters = dict(posterior.iloc[0])
# wg = bilby.gw.waveform_generator.WaveformGenerator(
#     duration=ifos.duration, sampling_frequency=ifos.sampling_frequency,
#     frequency_domain_source_model=memestr.core.waveforms.phenom.fd_imrx)
# wg_memory = bilby.gw.waveform_generator.WaveformGenerator(
#     duration=ifos.duration, sampling_frequency=ifos.sampling_frequency,
#     frequency_domain_source_model=memestr.core.waveforms.phenom.fd_imrx_with_memory)
#
# likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=wg)
# likelihood_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=wg_memory)
#
# likelihood.parameters = parameters
# likelihood_memory.parameters = parameters
#
# print(parameters['log_likelihood'])
# print(likelihood.log_likelihood_ratio())
# print(likelihood_memory.log_likelihood_ratio())
#
#
# reweighted_log_bf, log_weights = memestr.core.postprocessing.reweigh_by_likelihood(
#     new_likelihood=likelihood_memory, new_result=result, reference_likelihood=likelihood, reference_result=None)

# np.savetxt("{}_log_weights".format(event), log_weights)

log_bfs = []

for event in events:
    log_weights = np.loadtxt("{}_log_weights".format(event))
    reweighted_log_bf = logsumexp(log_weights) - np.log(len(log_weights))
    print(reweighted_log_bf)
    log_bfs.append(reweighted_log_bf)

gwtm_1_original = np.loadtxt("GWTM-1.txt")

plt.scatter(np.arange(0, 11), log_bfs, label="PhenomXHM")
plt.scatter(np.arange(0, 10), gwtm_1_original, label="NRHybSur")
plt.xticks(ticks=np.arange(0, 11), labels=events, rotation=75)
plt.legend()
plt.tight_layout()
plt.savefig("gwtm-1.png")
plt.clf()
