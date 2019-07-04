import matplotlib
import matplotlib.pyplot as plt

from bilby.gw.waveform_generator import WaveformGenerator

from memestr.core.waveforms import *

start_time = 0
duration = 16
sampling_frequency = 2048

wg_nr_gws = WaveformGenerator(
    frequency_domain_source_model=gws_nominal_hom,
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
    waveform_arguments=dict(alpha=0.1))
wg_nr_osc = WaveformGenerator(
    time_domain_source_model=time_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
    waveform_arguments=dict(alpha=0.1, minimum_frequency=10))
wg_nr_osc_mem = WaveformGenerator(
    time_domain_source_model=time_domain_nr_hyb_sur_waveform_with_memory,
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
    waveform_arguments=dict(alpha=0.1, minimum_frequency=10))
wg_imr_osc = WaveformGenerator(frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
                               sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
                               waveform_arguments=dict(alpha=0.1))
wg_nr_mem = WaveformGenerator(time_domain_source_model=time_domain_nr_hyb_sur_waveform_memory,
                              sampling_frequency=sampling_frequency, duration=duration, start_time=start_time,
                              waveform_arguments=dict(alpha=0.1, minimum_frequency=10))

mass_1 = 36.
mass_2 = 29.
luminosity_distance = 410.
chi_1 = -0.4
chi_2 = 0.4
theta_jn = 1.5
phase = 0.
psi = 0.
geocent_time = 0.
ra = 0.
dec = 0.

params = dict(mass_ratio=mass_2 / mass_1, total_mass=mass_1 + mass_2, s13=chi_1, s23=chi_2,
              luminosity_distance=luminosity_distance, inc=theta_jn, phase=phase, ra=ra, dec=dec,
              geocent_time=geocent_time, psi=psi)
params_gws = dict(mass_1=mass_1, mass_2=mass_2, chi_1=chi_1, chi_2=chi_2, luminosity_distance=luminosity_distance,
                  theta_jn=theta_jn, phase=phase - np.pi / 2, ra=ra, dec=dec, geocent_time=geocent_time, psi=psi)
font = {'family': 'sans-serif', 'weight': 300, 'size': 22}
matplotlib.rc('font', **font)
# matplotlib.rcParams.update({'text.usetex': True})
merger_index = np.argmax(wg_nr_osc_mem.time_domain_strain(params)['plus'])
merger_time = wg_nr_osc_mem.time_array[merger_index]
plt.plot(wg_nr_osc_mem.time_array - merger_time, 10**21*wg_nr_osc_mem.time_domain_strain(params)['plus'],
         label='Oscillatory + Memory')
plt.plot(wg_nr_mem.time_array - merger_time, 10**21*wg_nr_mem.time_domain_strain(params)['plus'], label='Memory', linestyle='--')
plt.plot([12.31 - merger_time, 12.31 - merger_time], [0, 10**21*np.max(wg_nr_mem.time_domain_strain(params)['plus'])], label='Memory displacement')
# plt.plot(wg_nr_gws.time_array, np.roll(wg_nr_gws.time_domain_strain(params_gws)['plus'], 16000))
plt.xlabel('Time [s]')
plt.ylabel('Strain [$10^{-21}$]')
plt.ylim(-0.95, 1.5)
plt.xlim(12.2 - merger_time, 12.32 - merger_time)
plt.legend()
plt.savefig('test_waveform')
plt.clf()

import sys
sys.exit(0)

# ifo_mem = bilby.gw.detector.get_empty_interferometer('H1')
# ifo_mem.minimum_frequency = 10
# ifo_mem.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, start_time=start_time, duration=duration)
# ifo_mem.inject_signal_from_waveform_generator(params, wg_nr_mem)
#
# ifo_osc = bilby.gw.detector.get_empty_interferometer('H1')
# ifo_osc.minimum_frequency = 10
# ifo_osc.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, start_time=start_time, duration=duration)
# ifo_osc.inject_signal_from_waveform_generator(params, wg_nr_osc)


# ifo_gws = bilby.gw.detector.get_empty_interferometer('H1')
# ifo_gws.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, start_time=start_time, duration=duration)
# ifo_gws.inject_signal_from_waveform_generator(params_gws, wg_nr_gws)

# ifo_imr = bilby.gw.detector.get_empty_interferometer('H1')
# ifo_imr.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, start_time=start_time, duration=duration)
# ifo_imr.inject_signal_from_waveform_generator(params, wg_imr_osc)
#
# plt.plot(ifo_osc.frequency_array, np.abs(ifo_osc.frequency_domain_strain))
# plt.plot(ifo_gws.frequency_array, np.abs(ifo_gws.frequency_domain_strain))
# plt.plot(ifo_imr.frequency_array, np.abs(ifo_imr.frequency_domain_strain))
# plt.plot(ifo_mem.frequency_array, np.abs(ifo_mem.frequency_domain_strain))
# plt.xlim(10, 1024)
# plt.loglog()
# plt.tight_layout()
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