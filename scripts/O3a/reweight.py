import pickle
import numpy as np
import sys
from scipy.special import logsumexp

import bilby

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
        # waveform_arguments = dict(minimum_frequency=0)  # VERY IMPORTANT
        waveform_arguments = dict(minimum_frequency=minimum_frequency, reference_frequency=20)  # VERY IMPORTANT
    oscillatory_model = memestr.waveforms.fd_nr_sur_7dq4
    memory_model = memestr.waveforms.fd_nr_sur_7dq4_with_memory
else:
    waveform_arguments = dict()
    oscillatory_model = memestr.waveforms.fd_imrx_fast
    memory_model = memestr.waveforms.fd_imrx_with_memory



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

outfile_name = f"{event}_memory_log_weights"
print(outfile_name)
if minimum_frequency != 0:
    outfile_name += f'_min_freq_{minimum_frequency}'
try:
    raise Exception
    log_memory_weights = np.loadtxt(outfile_name)
except Exception:
    use_stored_likelihood = minimum_frequency == 0
    reweighted_time_shift_memory_log_bf, log_memory_weights = memestr.postprocessing.reweight_by_likelihood_parallel(
        new_likelihood=likelihood_mem, result=result,
        reference_likelihood=likelihood_osc, use_stored_likelihood=use_stored_likelihood, n_parallel=n_parallel)
    np.savetxt(outfile_name, log_memory_weights)

reweighted_memory_log_bf = logsumexp(log_memory_weights) - np.log(len(log_memory_weights))
n_eff_memory = np.sum(np.exp(log_memory_weights)) ** 2 / np.sum(np.exp(log_memory_weights) ** 2)
print(n_eff_memory)
print(reweighted_memory_log_bf)
