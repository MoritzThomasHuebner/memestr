import bilby
import memestr

duration = 8.
sampling_frequency = 4096.

outdir = 'sanity_check'
label = 'GW150914'
bilby.core.utils.setup_logger(outdir=outdir, label=label)


# injection_parameters_memory = dict(
#     total_mass=70, mass_ratio=1.2, s11=0.0, s12=0.0, s13=0.0, s21=0.0,
#     s22=0.0, s23=0.0, luminosity_distance=500., inc=0.4, psi=2.659,
#     phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

# waveform_generator = bilby.gw.WaveformGenerator(
#     duration=duration, sampling_frequency=sampling_frequency,
#     time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
#     waveform_arguments=dict(alpha=0.1))

waveform_generator = bilby.gw.WaveformGenerator(
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    duration=duration,
    sampling_frequency=sampling_frequency,
    waveform_arguments=dict(alpha=0.1))
interferometers = bilby.gw.detector.get_event_data(label)
prior = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')
prior['a_1'] = bilby.gw.prior.DeltaFunction(0.0)
prior['a_2'] = bilby.gw.prior.DeltaFunction(0.0)
prior['tilt_1'] = bilby.gw.prior.DeltaFunction(0.0)
prior['tilt_2'] = bilby.gw.prior.DeltaFunction(0.0)
prior['phi_12'] = bilby.gw.prior.DeltaFunction(0.0)
prior['phi_jl'] = bilby.gw.prior.DeltaFunction(0.0)
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers, waveform_generator)

# ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
# ifos.set_strain_data_from_power_spectral_densities(
#     sampling_frequency=sampling_frequency, duration=duration,
#     start_time=injection_parameters_memory['geocent_time'] - 3)
# signal_ifos = ifos.inject_signal(waveform_generator=waveform_generator,
#                                  parameters=injection_parameters_memory)
#
# ifo = ifos[0]
# signal_ifo = ifo.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters_memory)
# ifo.plot_data(signal_ifo['cross'])
# print(signal_ifo)


result = bilby.run_sampler(likelihood, prior, sampler='dynesty',
                           outdir=outdir, label=label)
result.plot_corner()
