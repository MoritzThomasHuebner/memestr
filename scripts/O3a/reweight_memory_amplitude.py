import pickle
import sys

import bilby
import matplotlib.pyplot as plt
import numpy as np

import memestr
from memestr.events import events, precessing_events

event_number = int(sys.argv[1])
precessing = int(sys.argv[2]) == 1
n_parallel = int(sys.argv[3])
minimum_frequency = int(sys.argv[4])


if precessing:
    event_list = precessing_events
else:
    event_list = events
time_tag = event_list[event_number].time_tag
event = event_list[event_number].name

if precessing:
    if event == 'GW190521_prec':
        reference_frequency = 11
        waveform_arguments = dict(minimum_frequency=minimum_frequency, reference_frequency=reference_frequency)  # VERY IMPORTANT
    else:
        waveform_arguments = dict(minimum_frequency=minimum_frequency)  # VERY IMPORTANT
    oscillatory_model = memestr.waveforms.fd_nr_sur_7dq4
    memory_model = memestr.waveforms.fd_nr_sur_7dq4_with_memory
else:
    waveform_arguments = dict()
    oscillatory_model = memestr.waveforms.fd_imrx_fast
    memory_model = memestr.waveforms.fd_imrx_memory_only



if precessing:
    event += "_2000"
detectors = event_list[event_number].detectors
result = bilby.core.result.read_in_result(
    f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
result.outdir = f'{event}/result/'

print(len(result.posterior))
data_file = f'{event}/data/run_data0_{time_tag}_generation_data_dump.pickle'
print(data_file)
with open(data_file, "rb") as f:
    data_dump = pickle.load(f)
ifos = data_dump.interferometers
wg_osc = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=oscillatory_model,
    waveform_arguments=waveform_arguments)

wg_mem = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memory_model,
    waveform_arguments=waveform_arguments)

likelihood_osc = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_osc)
likelihood_mem = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_mem)

outfile_name = f"{event}_memory_amplitude_samples"

reweighting_terms = memestr.postprocessing.reconstruct_memory_amplitude_parallel(
    result=result, likelihood_memory=likelihood_mem, likelihood_oscillatory=likelihood_osc, n_parallel=n_parallel)
print(reweighting_terms)
print(reweighting_terms[0])
amplitude_samples = [term.memory_amplitude_sample for term in reweighting_terms]
d_inner_h_mem = [term.d_inner_h_mem for term in reweighting_terms]
optimal_snr_squared_h_mem = [term.optimal_snr_squared_h_mem for term in reweighting_terms]
h_osc_inner_h_mem = [term.h_osc_inner_h_mem for term in reweighting_terms]

np.savetxt(f'memory_amplitude_results/{event}_memory_amplitude_posterior.txt', amplitude_samples)
np.savetxt(f'memory_amplitude_results/{event}_d_inner_h_mem.txt', d_inner_h_mem)
np.savetxt(f'memory_amplitude_results/{event}_optimal_snr_squared_h_mem.txt', optimal_snr_squared_h_mem)
np.savetxt(f'memory_amplitude_results/{event}_h_osc_inner_h_mem.txt', h_osc_inner_h_mem)

plt.hist(amplitude_samples, bins='fd', density=True)
plt.xlabel('Memory amplitude')
plt.xlabel('p(Memory amplitude)')
plt.savefig(f'memory_amplitude_results/{event}_memory_amplitude_posterior.png')
