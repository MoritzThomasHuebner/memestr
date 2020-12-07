import pickle
import numpy as np
import sys
from scipy.special import logsumexp

import bilby

import memestr
from memestr.events import events

event_number = int(sys.argv[1])
# part = int(sys.argv[2])
# event_number = 0
time_tag = events[event_number].time_tag
event = events[event_number].name
detectors = events[event_number].detectors
# result = bilby.core.result.read_in_result(f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_par{part}_result.json')
result = bilby.core.result.read_in_result(f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
# result.outdir = f'{event}_{suffix}/result/'
result.outdir = f'{event}/result/'
result.plot_corner()
print(len(result.posterior))

# with open(f'{event}_{suffix}/data/run_data0_{time_tag}_generation_data_dump.pickle', "rb") as f:
with open(f'{event}/data/run_data0_{time_tag}_generation_data_dump.pickle', "rb") as f:
    data_dump = pickle.load(f)
ifos = data_dump.interferometers
bilby.core.utils.logger.disabled = True
wg_xhm_fast = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_imrx_fast)

wg_xhm = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_imrx)

wg_xhm_memory = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_imrx_with_memory)

bilby.core.utils.logger.disabled = False


likelihood_xhm_osc = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_xhm_fast)
likelihood_xhm_ref = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_xhm)
likelihood_xhm_memory = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_xhm_memory)

try:
    log_memory_weights = np.loadtxt(f"{event}_memory_log_weights")
    # log_memory_weights = np.loadtxt(f"{event}_memory_log_weights_par{part}")
except Exception:
    reweighted_time_shift_memory_log_bf, log_memory_weights = memestr.postprocessing.reweigh_by_likelihood(
        new_likelihood=likelihood_xhm_memory, result=result,
        reference_likelihood=likelihood_xhm_osc, use_stored_likelihood=True)
    np.savetxt(f"{event}_memory_log_weights", log_memory_weights)
    # np.savetxt(f"{event}_memory_log_weights_par{part}", log_memory_weights)

reweighted_memory_log_bf = logsumexp(log_memory_weights) - np.log(len(log_memory_weights))


n_eff_memory = np.sum(np.exp(log_memory_weights)) ** 2 / np.sum(np.exp(log_memory_weights) ** 2)
print(n_eff_memory)
print(reweighted_memory_log_bf)
