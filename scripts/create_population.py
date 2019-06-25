import logging
import numpy as np
import bilby
import sys
import matplotlib.pyplot as plt
import gwmemory

from memestr.core.population import generate_all_parameters
from memestr.core.waveforms import *
from memestr.core.parameters import AllSettings

logger = logging.getLogger('bilby')
logger.disabled = True

mass_kwargs = dict(alpha=1.5, beta=3, mmin=8, mmax=45)
all_params = generate_all_parameters(size=10000, clean=False, plot=False)

network_snrs = []
network_mem_snrs = []


def create_parameter_set(filename):
    best_snr = 0
    network_snr = 0
    settings = AllSettings()
    trials = 0
    while network_snr < 30:
        idx = np.random.randint(0, len(all_params.total_masses))
        total_mass = all_params.total_masses[idx]
        mass_ratio = all_params.mass_ratios[idx]
        if mass_ratio < 0.125:
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
        settings.waveform_data.sampling_frequency = 2048
        settings.waveform_data.duration = 16
        settings.waveform_arguments.l_max = 4
        waveform_generator_fd = \
            bilby.gw.WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
                                       parameters=settings.injection_parameters.__dict__,
                                       waveform_arguments=settings.waveform_arguments.__dict__,
                                       **settings.waveform_data.__dict__)

        # waveform_generator_memory = \
        #     bilby.gw.WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_memory_waveform_wrapped,
        #                                parameters=settings.injection_parameters.__dict__,
        #                                waveform_arguments=settings.waveform_arguments.__dict__,
        #                                **settings.waveform_data.__dict__)

        try:
            # hf_signal = waveform_generator.frequency_domain_strain()
            hf_signal = waveform_generator_fd.frequency_domain_strain()
            # hf_signal_mem = waveform_generator_memory.frequency_domain_strain()
            # hf_signal_mem_ref = waveform_generator_memory_ref.frequency_domain_strain()
        except ValueError as e:
            logger.warning(e)
            logger.info(str(settings.injection_parameters))
            continue

        ifos = bilby.gw.detector.InterferometerList([])
        # ifos_mem = bilby.gw.detector.InterferometerList([])
        # ifos_mem_ref = bilby.gw.detector.InterferometerList([])
        # for ifo in ['H1', 'L1', 'V1']:
        for ifo in ['H1', 'L1', 'V1']:
            start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
            interferometer = bilby.gw.detector.get_empty_interferometer(ifo)
            # interferometer_mem = bilby.gw.detector.get_empty_interferometer(ifo)
            # interferometer_mem_ref = bilby.gw.detector.get_empty_interferometer(ifo)
            if ifo in ['H1', 'L1']:
                interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
                # interferometer_mem.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
                # interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file('Aplus_asd.txt')
                # interferometer_mem.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file('Aplus_asd.txt')
            else:
                interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.\
                    from_power_spectral_density_file('AdV_psd.txt')
                # interferometer_mem.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.\
                #     from_power_spectral_density_file('AdV_psd.txt')
            interferometer.set_strain_data_from_power_spectral_density(
                sampling_frequency=settings.waveform_data.sampling_frequency,
                duration=settings.waveform_data.duration,
                start_time=start_time)
            # interferometer_mem.set_strain_data_from_power_spectral_density(
            #     sampling_frequency=settings.waveform_data.sampling_frequency,
            #     duration=settings.waveform_data.duration,
            #     start_time=start_time)

            injection_polarizations = interferometer.inject_signal(
                parameters=settings.injection_parameters.__dict__,
                injection_polarizations=hf_signal)
            # injection_polarizations_mem = interferometer_mem.inject_signal(
            #     parameters=settings.injection_parameters.__dict__,
            #     injection_polarizations=hf_signal_mem)

            # signal = interferometer.get_detector_response(
            #     injection_polarizations, settings.injection_parameters.__dict__)
            # signal_mem = interferometer_mem.get_detector_response(
            #     injection_polarizations_mem, settings.injection_parameters.__dict__)
            #
            # interferometer.plot_data(signal=signal, outdir='', label=str(filename) + '_osc')
            # interferometer_mem.plot_data(signal=signal_mem, outdir='.', label=str(filename) + '_mem')

            ifos.append(interferometer)
            # ifos_mem.append(interferometer_mem)
        best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
        best_snr = max(best_snrs)
        network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
        network_snrs.append(network_snr)
        best_mem_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
        best_mem_snr = max(best_mem_snrs)
        network_mem_snr = np.sqrt(np.sum([snr ** 2 for snr in best_mem_snrs]))
        network_mem_snrs.append(network_snr)
        trials += 1
    logger.disabled = False
    logger.info(filename)
    # logger.info(best_mem_snr)
    # logger.info(network_mem_snr)
    # logger.info(best_snr)
    # logger.info(network_snr)
    logger.disabled = True
    return ifos, settings.injection_parameters.__dict__, trials
    # with open('parameter_sets/' + str(filename), 'w') as f:
    #     f.write('total_mass=' + str(settings.injection_parameters.total_mass) +
    #             ' mass_ratio=' + str(settings.injection_parameters.mass_ratio) +
    #             ' luminosity_distance=' + str(settings.injection_parameters.luminosity_distance) +
    #             ' dec=' + str(settings.injection_parameters.dec) +
    #             ' ra=' + str(settings.injection_parameters.ra) +
    #             ' inc=' + str(settings.injection_parameters.inc) +
    #             ' psi=' + str(settings.injection_parameters.psi) +
    #             ' phase=' + str(settings.injection_parameters.phase) +
    #             ' geocent_time=' + str(settings.injection_parameters.geocent_time) +
    #             ' s11=' + str(settings.injection_parameters.s11) +
    #             ' s12=' + str(settings.injection_parameters.s12) +
    #             ' s13=' + str(settings.injection_parameters.s13) +
    #             ' s21=' + str(settings.injection_parameters.s21) +
    #             ' s22=' + str(settings.injection_parameters.s22) +
    #             ' s23=' + str(settings.injection_parameters.s23))

    # ifos.to_hdf5(outdir='parameter_sets', label=str(filename))
    # ifos_mem.to_hdf5(outdir='parameter_sets', label=str(filename))

output = 'Injection_log_bfs_' + str(sys.argv[1]) + '.txt'
with open(output, 'w') as f:
    f.write('# Memory Log BF\tTrials\n')

i = 0
while True:
# for i in range(int(sys.argv[1]), int(sys.argv[2])):
    settings =  AllSettings()
    ifos, injection_parameters, trials = create_parameter_set(i)
    settings.waveform_data.sampling_frequency = 2048
    settings.waveform_data.duration = 16

    waveform_generator_with_memory = \
        bilby.gw.WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
                                   parameters=injection_parameters,
                                   waveform_arguments=settings.waveform_arguments.__dict__,
                                   **settings.waveform_data.__dict__)
    waveform_generator_without_memory = \
        bilby.gw.WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
                                   parameters=injection_parameters,
                                   waveform_arguments=settings.waveform_arguments.__dict__,
                                   **settings.waveform_data.__dict__)
    likelihood_with_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                            waveform_generator=waveform_generator_with_memory)
    likelihood_without_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                               waveform_generator=waveform_generator_without_memory)
    likelihood_with_memory.parameters = injection_parameters
    likelihood_without_memory.parameters = injection_parameters
    res = likelihood_with_memory.log_likelihood_ratio() - likelihood_without_memory.log_likelihood_ratio()
    with open(output, 'a') as f:
        f.write(str(res) + '\t' + str(trials) + '\n')
    i += 1


# import matplotlib.pyplot as plt


# def read_snr(filename):
#     ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(filename) + '_H1L1V1.h5')
#     print(filename)
#     best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
#     network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
#     print(network_snr)
#     return network_snr


# for i in range(0, 2000):
#     network_snrs.append(read_snr(i))

# np.savetxt('network_snrs.txt', network_snrs)
# plt.hist(network_snrs, bins=int(np.sqrt(len(network_snrs))))
# plt.xlabel('Network SNR')
# plt.ylabel('Counts')
# plt.savefig('network_snrs.png')
# plt.clf()
# params = get_injection_parameter_set(id=10)
# print(params)
