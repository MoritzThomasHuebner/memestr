import os
import sys
import warnings
from bilby.core.utils import logger

logger.info('Test')

from memestr.core.parameters import AllSettings
from memestr.core.population import generate_all_parameters
from memestr.core.waveforms import *
from memestr.core.submit import get_injection_parameter_set

# logger.disabled = True
warnings.filterwarnings("ignore")

mass_kwargs = dict(alpha=1.5, beta=3, mmin=8, mmax=45)
logger.info('Generating population params')
all_params = generate_all_parameters(size=10000, clean=False, plot=False)
logger.info('Generated population params')

network_snrs = []


def create_parameter_set(filename, **kwargs):
    best_snr = 0
    network_snr = 0
    settings = AllSettings()
    trials = 0
    memory_log_bf = 0
    a = True
    while a:
        # while network_snr < 12 or best_snr < 8:
        idx = np.random.randint(0, len(all_params.total_masses))
        # total_mass = all_params.total_masses[idx]
        # mass_ratio = all_params.mass_ratios[idx]
        # if mass_ratio < 0.125:
        #     continue
        # luminosity_distance = np.random.choice(all_params.luminosity_distance)
        # dec = np.random.choice(all_params.dec)
        # ra = np.random.choice(all_params.ra)
        # inc = np.random.choice(all_params.inc)
        # psi = np.random.choice(all_params.psi)
        # phase = np.random.choice(all_params.phase)
        # geocent_time = np.random.choice(all_params.geocent_time)
        total_mass = 65
        mass_ratio = 0.8
        luminosity_distance = kwargs.get('luminosity_distance')
        dec = -1.2108
        ra = 1.375
        inc = 1.5
        psi = 2.659
        phase = 1.3
        geocent_time = 4
        s11 = 0.
        s12 = 0.
        # s13 = np.random.choice(all_params.s13)
        s13 = 0.
        s21 = 0.
        s22 = 0.
        s23 = 0.
        # s23 = np.random.choice(all_params.s23)

        settings.injection_parameters.update_args(mass_ratio=mass_ratio, total_mass=total_mass,
                                                  luminosity_distance=luminosity_distance, dec=dec, ra=ra,
                                                  inc=inc, psi=psi, phase=phase, geocent_time=geocent_time,
                                                  s11=s11, s12=s12, s13=s13,
                                                  s21=s21, s22=s22, s23=s23)
        settings.waveform_data.sampling_frequency = 2048
        settings.waveform_data.duration = 16
        settings.waveform_arguments.l_max = 4
        waveform_generator_with_memory = \
            bilby.gw.WaveformGenerator(
                time_domain_source_model=time_domain_IMRPhenomD_waveform_with_memory_wrapped,
                parameters=settings.injection_parameters.__dict__,
                waveform_arguments=settings.waveform_arguments.__dict__,
                **settings.waveform_data.__dict__)
        # waveform_generator_without_memory = \
        #     bilby.gw.WaveformGenerator(
        #         frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
        #         parameters=settings.injection_parameters.__dict__,
        #         waveform_arguments=settings.waveform_arguments.__dict__,
        #         **settings.waveform_data.__dict__)

        hf_signal = waveform_generator_with_memory.frequency_domain_strain()
        # hf_signal = waveform_generator_without_memory.frequency_domain_strain()

        ifos = bilby.gw.detector.InterferometerList([])
        for ifo in ['H1', 'L1', 'V1']:
            logger.disabled = True
            interferometer = setup_ifo(hf_signal, ifo, settings)
            logger.disabled = False
            ifos.append(interferometer)

        # likelihood_with_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
        #                                                                         waveform_generator=waveform_generator_with_memory)
        # likelihood_without_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
        #                                                                            waveform_generator=waveform_generator_without_memory)
        # likelihood_with_memory.parameters = settings.injection_parameters.__dict__
        # likelihood_without_memory.parameters = settings.injection_parameters.__dict__
        # memory_log_bf = likelihood_with_memory.log_likelihood_ratio() - likelihood_without_memory.log_likelihood_ratio()
        # logger.info('Memory log BF: ' + str(memory_log_bf))

        best_snrs = [ifo.meta_data['matched_filter_SNR'].real for ifo in ifos]
        best_snr = max(best_snrs)
        network_snr = np.sqrt(np.sum([snr ** 2 for snr in best_snrs]))
        network_snrs.append(network_snr)
        trials += 1
        # if os.path.exists('parameter_sets/' + str(filename)):
        #     return
        a = False
    logger.disabled = False
    # logger.info(filename)

    waveform_generator_memory = \
        bilby.gw.WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_memory_waveform_wrapped,
                                   parameters=settings.injection_parameters.__dict__,
                                   waveform_arguments=settings.waveform_arguments.__dict__,
                                   **settings.waveform_data.__dict__)
    hf_signal = waveform_generator_memory.frequency_domain_strain()
    mem_ifos = bilby.gw.detector.InterferometerList([])
    for ifo in ['H1', 'L1', 'V1']:
        logger.disabled = True
        interferometer = setup_ifo(hf_signal, ifo, settings)
        logger.disabled = False
        mem_ifos.append(interferometer)

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
    # ifos_mem.to_hdf5(outdir='parameter_sets', label=str(filename))
    return ifos, mem_ifos, settings.injection_parameters.__dict__, trials


def setup_ifo(hf_signal, ifo, settings):
    start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
    interferometer = bilby.gw.detector.get_empty_interferometer(ifo)
    if ifo in ['H1', 'L1']:
        interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        # interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_amplitude_spectral_density_file('Aplus_asd.txt')
    else:
        interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity. \
            from_power_spectral_density_file('AdV_psd.txt')
    interferometer.set_strain_data_from_power_spectral_density(
        sampling_frequency=settings.waveform_data.sampling_frequency,
        duration=settings.waveform_data.duration,
        start_time=start_time)
    injection_polarizations = interferometer.inject_signal(
        parameters=settings.injection_parameters.__dict__,
        injection_polarizations=hf_signal)
    return interferometer


# output = 'Injection_log_bfs/Injection_log_bfs_' + str('test') + '.txt'
# output = 'Injection_log_bfs/Injection_log_bfs_' + str(sys.argv[1]) + '.txt'
# logger.info(output)
# with open(output, 'w') as f:
#     f.write('# Memory Log BF\tTrials\tNetwork SNR\tMemory Network SNR\n')

# for i in range(int(sys.argv[1]), int(sys.argv[2])):
for i, ld in zip(['20000', '20001', '20002', '20003',
                  '20004', '20005', '20006', '20007',
                  '20008', '20009', '20010', '20011',
                  '20012', '20013', '20014', '20015',
                  '20016', '20017', '20018', '20019',
                  '20020', '20021', '20022', '20023',
                  '20024', '20025', '20026', '20027',
                  '20028', '20029'],
                 [2000.0, 1666.6666666666667, 1428.5714285714287, 1250.0,
                  1111.111111111111, 1000.0, 909.0909090909091, 833.3333333333334,
                  769.2307692307693, 714.2857142857143, 666.6666666666666, 625.0,
                  588.2352941176471, 555.5555555555555, 526.3157894736842, 500.0,
                  476.1904761904762, 454.54545454545456, 434.7826086956522, 416.6666666666667,
                  400.0, 384.61538461538464, 370.3703703703704, 357.14285714285717,
                  344.82758620689657, 333.3333333333333, 322.5806451612903, 312.5,
                  303.030303030303, 294.11764705882354]):
    create_parameter_set(i, luminosity_distance=ld)
sys.exit(0)
create_parameter_set(int(sys.argv[1]))

while True:
    logger.info('Start sampling population')
    settings = AllSettings()
    ifos, mem_ifos, injection_parameters, trials = create_parameter_set('')
    network_snr = np.sqrt(np.sum([ifo.meta_data['matched_filter_SNR'].real ** 2 for ifo in ifos]))
    memory_network_snr = np.sqrt(np.sum([ifo.meta_data['matched_filter_SNR'].real ** 2 for ifo in mem_ifos]))
    settings.waveform_data.sampling_frequency = 2048
    settings.waveform_data.duration = 16

    waveform_generator_with_memory = \
        bilby.gw.WaveformGenerator(
            frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
            parameters=injection_parameters,
            waveform_arguments=settings.waveform_arguments.__dict__,
            **settings.waveform_data.__dict__)
    waveform_generator_without_memory = \
        bilby.gw.WaveformGenerator(
            frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
            parameters=injection_parameters,
            waveform_arguments=settings.waveform_arguments.__dict__,
            **settings.waveform_data.__dict__)
    logger.disabled = True
    likelihood_with_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                            waveform_generator=waveform_generator_with_memory)
    likelihood_without_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                               waveform_generator=waveform_generator_without_memory)
    likelihood_with_memory.parameters = injection_parameters
    likelihood_without_memory.parameters = injection_parameters
    res = likelihood_with_memory.log_likelihood_ratio() - likelihood_without_memory.log_likelihood_ratio()
    logger.disabled = False
    logger.info(str(res) + '\t' + str(trials) + '\t' + str(network_snr) + '\t' + str(memory_network_snr))
    with open(output, 'a') as f:
        f.write(str(res) + '\t' + str(trials) + '\t' + str(network_snr) + '\t' + str(memory_network_snr) + '\n')

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
