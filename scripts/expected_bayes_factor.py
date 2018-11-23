import bilby as bb
import memestr as me
import numpy as np
import matplotlib.pyplot as plt

sampling_frequency = 4096
duration = 4
start_time = -2
waveform_generator = bb.gw.WaveformGenerator(
    time_domain_source_model=me.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time)
parameters = dict(total_mass=65., mass_ratio=1.2, s11=0.0, s12=0.0, s21=0.0, s22=0.0,
                  s13=0.0, s23=0.0, luminosity_distance=500., inc=1.5, psi=2.659,
                  phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_generator.parameters = parameters
hf_signal = waveform_generator.frequency_domain_strain()

ifos = [bb.gw.detector.get_interferometer_with_fake_noise_and_injection(name=name,
                                                                        injection_parameters=parameters,
                                                                        waveform_generator=waveform_generator,
                                                                        sampling_frequency=sampling_frequency,
                                                                        duration=duration, start_time=start_time,
                                                                        zero_noise=True) for name in ['H1', 'L1', 'V1']]
ifos = bb.gw.detector.InterferometerList(ifos)
ifos.inject_signal(parameters=parameters, waveform_generator=waveform_generator)

# for ifo in ifos:
#     print(ifo.optimal_snr_squared(signal=hf_signal))
injection_polarizations = waveform_generator.frequency_domain_strain(parameters)
distances = []
log_bayes_factors = []
for distance in range(100, 1000, 10):
    snr_tot_squared = 0
    for ifo in ifos:
        parameters['luminosity_distance'] = distance
        waveform_generator.parameters['luminosity_distance'] = distance
        signal_ifo = ifo.get_detector_response(injection_polarizations, parameters)
        ifo.strain_data.frequency_domain_strain = signal_ifo + ifo.strain_data.frequency_domain_strain
        opt_snr_squared = ifo.optimal_snr_squared(signal=signal_ifo).real
        # mf_snr = np.sqrt(ifo.matched_filter_snr_squared(signal=signal_ifo).real)
        # print(opt_snr_squared)
        # print(mf_snr)
        snr_tot_squared += opt_snr_squared
    log_bayes_factors.append(snr_tot_squared / 2)
    distances.append(distance)
plt.clf()
plt.plot(distances, log_bayes_factors)
plt.show()