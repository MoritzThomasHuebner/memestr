import matplotlib.pyplot as plt
import numpy as np

from memestr.waveforms.teobresums import *

mass_ratio = 0.8
total_mass = 200.
chi_1 = 0.
chi_2 = 0.
luminosity_distance = 500.
l_max = 2
phase = 0.1
inc = np.pi / 2
minimum_frequency = 20.
ecc = 0.9

psi = 1.
ra = 5.978302425373735
dec = 0.9149320164463342
geocent_time = 0.00

params = dict(mass_ratio=mass_ratio,
              total_mass=total_mass,
              chi_1=chi_1,
              chi_2=chi_2,
              luminosity_distance=luminosity_distance,
              l_max=l_max,
              phase=phase,
              inc=inc,
              minimum_frequency=minimum_frequency,
              ecc=ecc,
              psi=psi,
              ra=ra,
              dec=dec,
              geocent_time=geocent_time)

series = bilby.core.series.CoupledTimeAndFrequencySeries(duration=4, sampling_frequency=2048, start_time=0)
times = series.time_array
frequencies = series.frequency_array

waveform = td_teob_memory_only(times=times, mass_ratio=mass_ratio, total_mass=total_mass, chi_1=chi_1,
                               chi_2=chi_2, luminosity_distance=luminosity_distance, inc=inc, phase=phase, ecc=0,
                               modes=[[2, 2]], minimum_frequency=minimum_frequency)

plt.plot(times, waveform['plus'], label='e=0')
waveform = td_teob_memory_only(times=times, mass_ratio=mass_ratio, total_mass=total_mass, chi_1=chi_1,
                               chi_2=chi_2, luminosity_distance=luminosity_distance, inc=inc, phase=phase, ecc=ecc,
                               modes=[[2, 2]], minimum_frequency=minimum_frequency)

plt.plot(times, waveform['plus'], label=f'e={ecc}')
# plt.xlim(10, 1024)
plt.legend()
plt.savefig('example_teob.png')
plt.clf()

# wg = bilby.gw.waveform_generator.WaveformGenerator(duration=series.duration,
#                                                    sampling_frequency=series.sampling_frequency,
#                                                    frequency_domain_source_model=fd_teob_memory_only)
# snrs = []
# eccs = np.linspace(0, 0.99, 100)
#
# for ecc in eccs:
#     # print(ecc)
#     params['ecc'] = ecc
#     wg.parameters = params
#     bilby.core.utils.logger.disabled = True
#     ifo = bilby.gw.detector.get_empty_interferometer('H1')
#     ifo.set_strain_data_from_zero_noise(sampling_frequency=series.sampling_frequency, duration=series.duration,
#                                         start_time=series.start_time)
#     ifo.inject_signal_from_waveform_generator(parameters=params, waveform_generator=wg)
#     snrs.append(ifo.meta_data['optimal_SNR'])
#     bilby.core.utils.logger.disabled = False
#     print(ifo.meta_data['optimal_SNR'])
#
#
# plt.plot(eccs, snrs)
# plt.xlabel("e")
# plt.ylabel("SNR")
# plt.savefig('snr_vs_ecc.png')
# plt.clf()
