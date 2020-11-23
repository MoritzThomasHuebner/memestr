import warnings

import pandas as pd
from bilby.core.utils import logger

from memestr.population import generate_all_parameters
from memestr.waveforms import *

warnings.filterwarnings("ignore")
mass_kwargs = dict(alpha=1.5, beta=3, mmin=8, mmax=45)
logger.info('Generating population parameters')
# all_params = generate_all_parameters(size=10000, clean=False, plot=False, **mass_kwargs)
all_params = dict()
logger.info('Generated population parameters')

network_snrs = []


def create_injection():
    best_snr = 0
    network_snr = 0
    params = dict()

    while network_snr < 12 or best_snr < 8:
        idx = np.random.randint(0, len(all_params.total_masses))
        total_mass = all_params.total_masses[idx]
        mass_ratio = all_params.mass_ratios[idx]
        luminosity_distance = np.random.choice(all_params.luminosity_distance)
        dec = np.random.choice(all_params.dec)
        ra = np.random.choice(all_params.ra)
        inc = np.random.choice(all_params.inc)
        psi = np.random.choice(all_params.psi)
        phase = np.random.choice(all_params.phase)
        geocent_time = np.random.choice(all_params.geocent_time)

        s11 = 0.
        s12 = 0.
        s13 = np.random.choice(all_params.s13)
        s21 = 0.
        s22 = 0.
        s23 = np.random.choice(all_params.s23)

        injection_parameters = dict(mass_ratio=mass_ratio, total_mass=total_mass,
                                    luminosity_distance=luminosity_distance, dec=dec, ra=ra,
                                    inc=inc, psi=psi, phase=phase, geocent_time=geocent_time,
                                    s11=s11, s12=s12, s13=s13, s21=s21, s22=s22, s23=s23)
        sampling_frequency = 2048
        duration = 16

        logger.disabled = True
        waveform_generator = \
            bilby.gw.WaveformGenerator(
                frequency_domain_source_model=fd_imrx, parameters=injection_parameters,
                sampling_frequency=sampling_frequency, duration=duration, start_time=0)

        ifos = bilby.gw.detector.InterferometerList([])
        for ifo in ['H1', 'L1', 'V1']:
            interferometer = setup_ifo(waveform_generator, ifo, geocent_time,
                                       injection_parameters, zero_noise=False, aplus=False)
            ifos.append(interferometer)
        logger.disabled = False

        best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
        best_snr = max(best_snrs)
        network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))

        params = pd.DataFrame({'total_mass': [all_params.total_masses[idx]],
                               'mass_ratio': [all_params.mass_ratios[idx]],
                               'luminosity_distance': [np.random.choice(all_params.luminosity_distance)],
                               'dec': [np.random.choice(all_params.dec)],
                               'ra': [np.random.choice(all_params.ra)],
                               'inc': [np.random.choice(all_params.inc)],
                               'psi': [np.random.choice(all_params.psi)],
                               'phase': [np.random.choice(all_params.phase)],
                               'geocent_time': [np.random.choice(all_params.geocent_time)],
                               's13': [np.random.choice(all_params.s13)],
                               's23': [np.random.choice(all_params.s23)]})
    return params


def setup_ifo(waveform_generator, ifo, geocent_time,
              injection_parameters, zero_noise=False, aplus=False):
    start_time = geocent_time + 2 - waveform_generator.duration
    interferometer = bilby.gw.detector.get_empty_interferometer(ifo)
    if ifo in ['H1', 'L1']:
        if aplus:
            interferometer.power_spectral_density = \
                bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file('Aplus_asd.txt')
        else:
            interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
    else:
        interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity. \
            from_power_spectral_density_file('AdV_psd.txt')
    if zero_noise:
        interferometer.set_strain_data_from_zero_noise(
            sampling_frequency=waveform_generator.sampling_frequency,
            duration=waveform_generator.duration,
            start_time=start_time)
    else:
        interferometer.set_strain_data_from_power_spectral_density(
            sampling_frequency=waveform_generator.sampling_frequency,
            duration=waveform_generator.duration,
            start_time=start_time)
    _ = interferometer.inject_signal(
        parameters=injection_parameters,
        waveform_generator=waveform_generator)
    return interferometer