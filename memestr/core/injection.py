import json
import warnings

import pandas as pd
from bilby.core.utils import logger

from memestr.core.parameters import AllSettings
from memestr.core.population import generate_all_parameters, setup_ifo
from memestr.core.waveforms import *

warnings.filterwarnings("ignore")
mass_kwargs = dict(alpha=1.5, beta=3, mmin=8, mmax=45)
logger.info('Generating population params')
all_params = generate_all_parameters(size=10000, clean=False, plot=False)
logger.info('Generated population params')

network_snrs = []


def create_injection(**kwargs):
    best_snr = 0
    network_snr = 0
    settings = AllSettings.from_defaults_with_some_specified_kwargs(**kwargs)
    trials = 0

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

        settings.injection_parameters.update_args(mass_ratio=mass_ratio, total_mass=total_mass,
                                                  luminosity_distance=luminosity_distance, dec=dec, ra=ra,
                                                  inc=inc, psi=psi, phase=phase, geocent_time=geocent_time,
                                                  s11=s11, s12=s12, s13=s13,
                                                  s21=s21, s22=s22, s23=s23)
        settings.waveform_data.sampling_frequency = 2048
        settings.waveform_data.duration = 16
        settings.waveform_arguments.l_max = 4

        logger.disabled = True
        waveform_generator = \
            bilby.gw.WaveformGenerator(
                frequency_domain_source_model=fd_imrx,
                parameters=settings.injection_parameters.__dict__,
                waveform_arguments=settings.waveform_arguments.__dict__,
                **settings.waveform_data.__dict__)
        logger.disabled = False

        hf_signal = waveform_generator.frequency_domain_strain()
        ifos = bilby.gw.detector.InterferometerList([])
        for ifo in ['H1', 'L1', 'V1']:
            logger.disabled = True
            interferometer = setup_ifo(hf_signal, ifo, settings, aplus=False)
            logger.disabled = False
            ifos.append(interferometer)

        best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
        best_snr = max(best_snrs)
        network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
        network_snrs.append(network_snr)
        print(network_snr)
        print(best_snr)
        trials += 1
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
