import pickle

import bilby
import matplotlib.pyplot as plt

import memestr
from memestr.events import events
from memestr.postprocessing import reweight_by_memory_amplitude

event_number = 0  # int(sys.argv[1])
precessing = 0  # int(sys.argv[2]) == 1
minimum_frequency = 0  # int(sys.argv[3])

event_list = events
time_tag = event_list[event_number].time_tag
event = event_list[event_number].name


detectors = event_list[event_number].detectors
result = bilby.core.result.read_in_result(
    f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
result.outdir = f'{event}/result/'

data_file = f'{event}/data/run_data0_{time_tag}_generation_data_dump.pickle'

with open(data_file, "rb") as f:
    data_dump = pickle.load(f)
ifos = data_dump.interferometers
wg_osc = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_imrx_fast)

wg_mem = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_imrx_memory_only)

wg_comb = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_imrx_with_memory)

likelihood_osc = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_osc)
likelihood_mem = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_mem)
likelihood_comb = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_comb)

params = dict(result.posterior.iloc[-1])
memory_amplitude = 5
params['memory_amplitude'] = memory_amplitude
likelihood_osc.parameters = params
likelihood_mem.parameters = params
likelihood_comb.parameters = params
plt.plot(likelihood_osc.waveform_generator.time_array, likelihood_osc.waveform_generator.time_domain_strain(parameters=params)['plus'])
plt.plot(likelihood_comb.waveform_generator.time_array, likelihood_comb.waveform_generator.time_domain_strain(parameters=params)['plus'])
plt.xlim(likelihood_comb.waveform_generator.time_array[-300], likelihood_comb.waveform_generator.time_array[-1])
plt.savefig('test_plus_local.png')
plt.clf()

plt.plot(likelihood_osc.waveform_generator.time_array, likelihood_osc.waveform_generator.time_domain_strain(parameters=params)['cross'])
plt.plot(likelihood_comb.waveform_generator.time_array, likelihood_comb.waveform_generator.time_domain_strain(parameters=params)['cross'])
plt.xlim(likelihood_comb.waveform_generator.time_array[-300], likelihood_comb.waveform_generator.time_array[-1])
plt.savefig('test_cross_local.png')
plt.clf()

ma = memestr.postprocessing.MemoryAmplitudeReweighter(likelihood_oscillatory=likelihood_osc,
                                                      likelihood_memory=likelihood_mem)
ma.calculate_reweighting_terms(parameters=params)
log_l_reweighted = ma.reweight_by_memory_amplitude(memory_amplitude=1)
# log_l_reweighted = reweight_by_memory_amplitude(
#     memory_amplitude=memory_amplitude, d_inner_h_mem=ma.d_inner_h_mem,
#     optimal_snr_squared_h_mem=ma.optimal_snr_squared_h_mem, h_osc_inner_h_mem=ma.h_osc_inner_h_mem)
log_l_osc = likelihood_osc.log_likelihood_ratio()
log_l_mem = likelihood_mem.log_likelihood_ratio()
log_l_comb = likelihood_comb.log_likelihood_ratio()

print(params['log_likelihood'])
print(log_l_osc)
print()
print(log_l_reweighted + log_l_osc)
print(log_l_comb)
