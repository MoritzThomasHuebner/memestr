import bilby
import matplotlib.pyplot as plt
import numpy as np

from memestr.core.postprocessing import *

wf = 'IMRPhenomPv2'
injection = {
    'chi_2': -0.5, 'chi_1': 0.2, 'theta_jn': 1.0,
    'psi': 1.8616810188688206, 'phase': 5.1380191890917972,
    'mass_2': 10.13750619740501, 'mass_1': 24.775312396456464,
    'luminosity_distance': 1050.2069895665365, 'dec': -1.2070840471049264,
    'ra': 2.155457582628403, 'geocent_time': 1126259642.414596}

duration = 4.0
sampling_frequency = 2048.0
freqs = np.linspace(
    0., sampling_frequency / 2, sampling_frequency / 2 * duration + 1)
times = np.arange(0, duration, 1. / sampling_frequency)

waveform_arguments_22 = dict(
    waveform_approximant=wf, reference_frequency=50.,
    minimum_frequency=20.)

waveform_generator_22 = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments_22,
    start_time=injection['geocent_time'] - 3)

hf_22 = waveform_generator_22.frequency_domain_strain(parameters=injection)
strain_22 = hf_22['plus'] + 1j * hf_22['cross']

waveform_arguments_HM = dict(reference_frequency=50.,
                             minimum_frequency=20.,
                             comparison_waveform=wf, return_correction=True)

waveform_generator_HM_over = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=gws_overlap,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments_HM,
    start_time=injection['geocent_time'] - 3)

hf_HM_over = waveform_generator_HM_over.frequency_domain_strain(parameters=injection)
strain_HM_over = hf_HM_over['plus'] + 1j * hf_HM_over['cross']

print('outside overlap: ', overlap_function(hf_HM_over, hf_22, freqs))

# replace with the appropriate time and psi here below:
new_geocent_time = injection['geocent_time'] + hf_HM_over['t_shift']
new_phi = hf_HM_over['phase_new']
print('new time: {}'.format(new_geocent_time))
print('new phi : {}'.format(new_phi))

#############################################################################

injection2 = injection.copy()
injection2['phase'] = new_phi
injection2['geocent_time'] = new_geocent_time

waveform_generator_HM = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=gws_nominal,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments_HM,
    start_time=injection['geocent_time'] - 3)

hf_HM = waveform_generator_HM.frequency_domain_strain(parameters=injection2)
strain_HM = hf_HM['plus'] + 1j * hf_HM['cross']
strain_HM *= np.exp(-2j * np.pi * hf_HM_over['t_shift'] * freqs)
# strain_HM *= np.exp(-1j*new_phi)

f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                             gridspec_kw={'hspace': 0},
                             figsize=(8, 6))

ax1.loglog(freqs, np.abs(strain_HM), label='NR from new inj')
ax1.loglog(freqs, np.abs(strain_22), label='22 Pv2')
ax1.loglog(freqs, np.abs(strain_HM_over), label='NR from overlap')
ax2.semilogx(freqs, np.angle(strain_HM))
ax2.semilogx(freqs, np.angle(strain_22))
ax2.semilogx(freqs, np.angle(strain_HM_over))
ax1.set_ylabel(r'$|\tilde{h}(f)|$')
ax2.set_ylabel(r'$\theta$')
ax2.set_xlabel(r'$f$')
# ax1.set_ylim(1e-26,1e-21)
ax2.set_xlim(1e1, )
plt.tight_layout()
ax1.legend()
f.savefig('FD_source.png')

waveform_arguments_HM = dict(reference_frequency=50.,
                             minimum_frequency=20.,
                             comparison_waveform=wf, return_correction=False)
waveform_generator_HM_over = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=gws_overlap,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments_HM,
    start_time=injection['geocent_time'] - 3)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
# ifos.set_strain_data_from_zero_noise(
#    duration=duration, sampling_frequency=sampling_frequency)
for ifo in ifos:
    ifo.set_strain_data_from_zero_noise(sampling_frequency, duration, start_time=injection['geocent_time'] - 3)
    ifo.strain_data.start_time = injection['geocent_time'] - 3
inj = {key: hf_HM_over[key] for key in ['plus', 'cross']}
ifos.inject_signal(
    injection_polarizations=inj, parameters=injection)

likelihood_22 = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_22)
likelihood_22.parameters.update(injection)

likelihood_HM_over = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_HM_over)
likelihood_HM_over.parameters.update(injection)

print('22 likelihood        : {}'.format(likelihood_22.log_likelihood_ratio()))
print('HM overlap likelihood: {}'.format(likelihood_HM_over.log_likelihood_ratio()))

likelihood_HM = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_HM)
likelihood_HM.parameters.update(injection2)

print('HM likelihood        : {}'.format(likelihood_HM.log_likelihood_ratio()))
