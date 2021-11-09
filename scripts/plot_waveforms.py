import matplotlib.pyplot as plt
import numpy as np

import bilby
from bilby.gw.waveform_generator import WaveformGenerator

from memestr.waveforms import td_imrx_with_memory, td_imrx_memory_only

plt.style.use('paper.mplstyle')

start_time = 0
duration = 4
sampling_frequency = 2048

wg_imr_mem = WaveformGenerator(time_domain_source_model=td_imrx_memory_only,
                               sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
                               waveform_arguments=dict(alpha=0.1))
wg_imr_osc = WaveformGenerator(time_domain_source_model=td_imrx_with_memory,
                               sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
                               waveform_arguments=dict(alpha=0.1))


mass_1 = 36.
mass_2 = 29.
luminosity_distance = 410.
chi_1 = 0.2
chi_2 = 0.3
theta_jn = 1.5
phase = 0.
psi = 0.
geocent_time = 0.
ra = 0.
dec = 0.

params = dict(mass_ratio=mass_2 / mass_1, total_mass=mass_1 + mass_2, s13=chi_1, s23=chi_2,
              luminosity_distance=luminosity_distance, inc=theta_jn, phase=phase, ra=ra, dec=dec,
              geocent_time=geocent_time, psi=psi)

wg_imr_mem.parameters = params
wg_imr_osc.parameters = params

plt.plot(wg_imr_osc.time_array, wg_imr_osc.time_domain_strain()['plus'], label='$h_{\mathrm{osc+mem}}(t)$')
plt.plot(wg_imr_mem.time_array, wg_imr_mem.time_domain_strain()['plus'], label='$h_{\mathrm{mem}}(t)$')
plt.xlabel('time [s]')
plt.ylabel('strain')
plt.xlim(1.6, 1.85)
plt.legend()
plt.tight_layout()
plt.savefig('test.pdf')
plt.clf()

# for mass_ratio in [0.125, 0.25, 0.5, 0.75, 1.]:
#     # params['total_mass'] = total_mass
#     params['mass_ratio'] = mass_ratio
#     ifo_mem = bilby.gw.detector.get_empty_interferometer('H1')
#     ifo_mem.minimum_frequency = 1
#     ifo_mem.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, start_time=start_time, duration=duration)
#     _ = ifo_mem.inject_signal_from_waveform_generator(params, wg_imr_mem)
#
#     ifo_osc = bilby.gw.detector.get_empty_interferometer('H1')
#     ifo_osc.minimum_frequency = 1
#     ifo_osc.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, start_time=start_time, duration=duration)
#     _ = ifo_osc.inject_signal_from_waveform_generator(params, wg_imr_osc)
#
#     snr = ifo_osc.meta_data['optimal_SNR']
#     plt.plot(ifo_mem.frequency_array, np.abs(ifo_mem.frequency_domain_strain), label=str(mass_ratio))
#     print(snr)
#     # plt.plot(ifo_mem.frequency_array, np.abs(ifo_mem.frequency_domain_strain)/snr, label=str(total_mass))
# # plt.plot(ifo_mem.power_spectral_density.frequency_array, ifo_mem.power_spectral_density.asd_array, label='H1 ASD')
# # plt.plot([1, 1000], [1e-24, 1e-27], label='1/x')
# plt.xlim(20, 1000)
# # plt.ylim(1e-26, 1e-24)
# plt.loglog()
# plt.legend()
# plt.tight_layout()
# plt.savefig('different_masses')
# plt.show()
# plt.clf()

# from bilby.gw import utils as gwutils

# signal_ifo = ifo_osc.get_detector_response(wg_nr_osc.frequency_domain_strain(), parameters=params)

# df = ifo_osc.strain_data.frequency_array[1] - ifo_osc.strain_data.frequency_array[0]
# asd_osc = gwutils.asd_from_freq_series(
#     freq_data=ifo_osc.strain_data.frequency_domain_strain, df=df)
# asd_mem = gwutils.asd_from_freq_series(
#     freq_data=ifo_mem.strain_data.frequency_domain_strain, df=df)
#
# plt.loglog(ifo_osc.strain_data.frequency_array,
#           asd_osc,
#           color='C0', label='Oscillatory signal')
# plt.loglog(ifo_mem.strain_data.frequency_array,
#           asd_mem,
#           color='C1', label='Memory signal')
# plt.loglog(ifo_osc.strain_data.frequency_array,
#           ifo_osc.amplitude_spectral_density_array,
#           color='C2', lw=1.0, label=ifo_osc.name + ' ASD')
#
# signal_asd = gwutils.asd_from_freq_series(
#     freq_data=signal_ifo, df=df)
#
# plt.grid(True)
# plt.xlim(10, 1024)
# plt.ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
# plt.xlabel(r'Frequency [Hz]')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
# plt.clf()
#