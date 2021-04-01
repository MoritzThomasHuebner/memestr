import pickle
import numpy as np
import sys
from scipy.special import logsumexp

import bilby

import memestr
from memestr.events import events, precessing_events

event_number = int(sys.argv[1])
precessing = int(sys.argv[2]) == 1

minimum_frequency = 20
if precessing:
    event_list = precessing_events
    # waveform_arguments = dict(minimum_frequency=0)  # VERY IMPORTANT
    waveform_arguments = dict(minimum_frequency=minimum_frequency)  # VERY IMPORTANT
    oscillatory_model = memestr.waveforms.fd_nr_sur_7dq4
    memory_model = memestr.waveforms.fd_nr_sur_7dq4_with_memory
else:
    event_list = events
    waveform_arguments = dict()
    oscillatory_model = memestr.waveforms.fd_imrx_fast
    memory_model = memestr.waveforms.fd_imrx_with_memory

time_tag = event_list[event_number].time_tag
event = event_list[event_number].name
if precessing:
    event += "_2000"
detectors = event_list[event_number].detectors
result = bilby.core.result.read_in_result(f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
result.outdir = f'{event}/result/'
result.plot_corner()
print(len(result.posterior))
with open(f'{event}/data/run_data0_{time_tag}_generation_data_dump.pickle', "rb") as f:
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

outfile_name = f"{event}_memory_log_weights"
if minimum_frequency == 20:
    outfile_name += '_min_freq_20'
try:
    log_memory_weights = np.loadtxt(outfile_name)
except Exception:
    reweighted_time_shift_memory_log_bf, log_memory_weights = memestr.postprocessing.reweight_by_likelihood_parallel(
        new_likelihood=likelihood_mem, result=result,
        reference_likelihood=likelihood_osc, use_stored_likelihood=True, n_parallel=8)
    np.savetxt(outfile_name, log_memory_weights)

reweighted_memory_log_bf = logsumexp(log_memory_weights) - np.log(len(log_memory_weights))
n_eff_memory = np.sum(np.exp(log_memory_weights)) ** 2 / np.sum(np.exp(log_memory_weights) ** 2)
print(n_eff_memory)
print(reweighted_memory_log_bf)
