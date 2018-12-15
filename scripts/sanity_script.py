import bilby
import numpy as np
import memestr

duration = 8.
sampling_frequency = 2048.

outdir = 'sanity_check'
label = 'res'
bilby.core.utils.setup_logger(outdir=outdir, label=label)


injection_parameters_memory = dict(
    total_mass=70, mass_ratio=1.2, s11=0.0, s12=0.0, s13=0.0, s21=0.0,
    s22=0.0, s23=0.0, luminosity_distance=500., inc=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments_imr = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=50., minimum_frequency=20.)

waveform_generator_memory = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    waveform_arguments=dict(alpha=0.1))

waveform_generator_imr = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments_imr)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters_memory['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator_memory,
                   parameters=injection_parameters_memory)

priors = bilby.gw.prior.PriorDict()
priors['mass_1'] = injection_parameters_memory['total_mass'] * injection_parameters_memory['mass_ratio']
priors['mass_2'] = injection_parameters_memory['total_mass'] - priors['mass_1']
priors['luminosity_distance'] = injection_parameters_memory['luminosity_distance']
priors['iota'] = injection_parameters_memory['inc']
priors['phase'] = injection_parameters_memory['phase']
priors['ra'] = injection_parameters_memory['ra']
priors['dec'] = injection_parameters_memory['dec']
priors['psi'] = injection_parameters_memory['psi']
priors['geocent_time'] = bilby.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time')
priors['a_1'] = 0
priors['a_2'] = 0
priors['tilt_1'] = 0
priors['tilt_2'] = 0
priors['phi_12'] = 0
priors['phi_jl'] = 0

priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters_memory['geocent_time'] - 1,
    maximum=injection_parameters_memory['geocent_time'] + 1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_imr)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000, outdir=outdir, label=label)

result.plot_corner()
