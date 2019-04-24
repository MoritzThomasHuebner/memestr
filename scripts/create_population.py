import logging
import numpy as np
import bilby
import sys

from memestr.core.population import generate_all_parameters
from memestr.core.waveforms import time_domain_nr_hyb_sur_waveform_with_memory_wrapped
from memestr.core.parameters import AllSettings

logger = logging.getLogger('bilby')
logger.disabled = True

mass_kwargs = dict(alpha=1.5, beta=3, mmin=8, mmax=45)
all_params = generate_all_parameters(size=10000, clean=False, plot=False)

network_snrs = []


def create_parameter_set(filename):
    best_snr = 0
    network_snr = 0
    settings = AllSettings()
    while best_snr < 8 and network_snr < 12:
        idx = np.random.randint(0, len(all_params.total_masses))
        total_mass = all_params.total_masses[idx]
        if total_mass < 15:
            continue
        mass_ratio = all_params.mass_ratios[idx]
        if mass_ratio < 0.5:
            continue
        luminosity_distance = np.random.choice(all_params.luminosity_distance)
        dec = np.random.choice(all_params.dec)
        ra = np.random.choice(all_params.ra)
        inc = np.random.choice(all_params.inc)
        psi = np.random.choice(all_params.psi)
        phase = np.random.choice(all_params.phase)
        geocent_time = np.random.choice(all_params.geocent_time)
        s11 = 0
        s12 = 0
        s13 = np.random.choice(all_params.s13)
        s21 = 0
        s22 = 0
        s23 = np.random.choice(all_params.s23)

        settings.injection_parameters.update_args(mass_ratio=mass_ratio, total_mass=total_mass,
                                                  luminosity_distance=luminosity_distance, dec=dec, ra=ra,
                                                  inc=inc, psi=psi, phase=phase, geocent_time=geocent_time,
                                                  s11=s11, s12=s12, s13=s13,
                                                  s21=s21, s22=s22, s23=s23)
        settings.waveform_data.duration = 16
        settings.waveform_arguments.l_max = 4
        waveform_generator = \
            bilby.gw.WaveformGenerator(time_domain_source_model=time_domain_nr_hyb_sur_waveform_with_memory_wrapped,
                                       parameters=settings.injection_parameters.__dict__,
                                       waveform_arguments=settings.waveform_arguments.__dict__,
                                       **settings.waveform_data.__dict__)
        try:
            hf_signal = waveform_generator.frequency_domain_strain()
        except ValueError:
            print(settings.injection_parameters)
            continue
        ifos = [bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
            name,
            injection_polarizations=hf_signal,
            injection_parameters=settings.injection_parameters.__dict__,
            zero_noise=False,
            plot=False,
            **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors]
        ifos = bilby.gw.detector.InterferometerList(ifos)
        best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
        best_snr = max(best_snrs)
        network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
        network_snrs.append(network_snr)
    print(best_snr)
    print(network_snr)
    print(filename)
    print('\n')

    with open('parameter_sets/' + str(filename), 'w') as f:
        f.write('total_mass=' + str(settings.injection_parameters.total_mass) +
                ' mass_ratio=' + str(settings.injection_parameters.mass_ratio) +
                ' luminosity_distance=' + str(settings.injection_parameters.luminosity_distance) +
                ' dec=' + str(settings.injection_parameters.dec) +
                ' ra=' + str(settings.injection_parameters.ra) +
                ' inc=' + str(settings.injection_parameters.inc) +
                ' psi=' + str(settings.injection_parameters.psi) +
                ' phase=' + str(settings.injection_parameters.phase) +
                ' geocent_time=' + str(settings.injection_parameters.geocent_time) +
                ' s11=' + str(settings.injection_parameters.s11) +
                ' s12=' + str(settings.injection_parameters.s12) +
                ' s13=' + str(settings.injection_parameters.s13) +
                ' s21=' + str(settings.injection_parameters.s21) +
                ' s22=' + str(settings.injection_parameters.s22) +
                ' s23=' + str(settings.injection_parameters.s23))
    ifos.to_hdf5(outdir='parameter_sets', label=str(filename))


# for i in range(int(sys.argv[1]), int(sys.argv[2])):
for i in range(100):
    create_parameter_set(i)

import matplotlib.pyplot as plt


def read_snr(filename):
    ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(filename) + '_H1L1V1.h5')
    print(filename)
    best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
    network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
    print(network_snr)
    return network_snr


# for i in range(0, 1000):
#     network_snrs.append(read_snr(i))


# plt.hist(network_snrs, bins=int(np.sqrt(len(network_snrs))))
# plt.xlabel('Network SNR')
# plt.ylabel('Counts')
# plt.savefig('network_snrs.png')
# plt.clf()
# params = get_injection_parameter_set(id=10)
# print(params)
