import bilby
import numpy as np
import memestr
from memestr.core.waveforms import *
from copy import deepcopy

logger = bilby.core.utils.logger

data = np.genfromtxt('GW150914/time_data.dat')
time_of_event = data[0]
start_time = data[1]
duration = data[2]
minimum_frequency = data[3]
sampling_frequency = data[4]

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])

for name, ifo in zip(['H1', 'L1'], ifos):
    psd = bilby.gw.detector.psd.PowerSpectralDensity.from_amplitude_spectral_density_file('GW150914/' + name + '_psd.dat')
    ifo.power_spectral_density = psd
    strain = np.loadtxt('GW150914/' + name + '_frequency_domain_data.dat')
    strain = strain[:, 1] + 1j*strain[:, 2]
    ifo.set_strain_data_from_frequency_domain_strain(strain, sampling_frequency=sampling_frequency,
                                                     duration=duration, start_time=start_time)
    ifo.power_spectral_density.psd_array = np.minimum(ifo.power_spectral_density.psd_array, 1)

hom_result = bilby.result.read_in_result(filename='GW150914/corrected_result.json')
base_result = bilby.result.read_in_result(filename='GW150914/22_pe_result.json')

waveform_generator_imr = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
    start_time=start_time, duration=duration, sampling_frequency=sampling_frequency)

waveform_generator_hom = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
    duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)

waveform_generator_memory = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
    duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)

likelihood_imr_phenom = bilby.gw.likelihood \
    .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                waveform_generator=waveform_generator_imr)

likelihood_hom = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_hom)


likelihood_memory = bilby.gw.likelihood \
    .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                waveform_generator=waveform_generator_memory)

likelihoods_22 = hom_result.posterior['log_likelihood']
posterior_dict_22 = deepcopy(base_result.posterior)
posterior_dict_hom = deepcopy(hom_result.posterior)
number_of_samples = len(likelihoods_22)

likelihoods_hom = []
weights = []


ethan_result = np.loadtxt('GW150914/new_likelihoods.dat')
ethan_22_log_likelihood = ethan_result[:, 0]
ethan_hom_log_likelihood = ethan_result[:, 1]
ethan_weight_log_likelihood = ethan_result[:, 2]

for i in range(len(posterior_dict_22)):

    likelihood_imr_parameters = dict(
        total_mass=posterior_dict_22['mass_1'][i]+posterior_dict_22['mass_2'][i],
        mass_ratio=posterior_dict_22['mass_2'][i]/posterior_dict_22['mass_1'][i],
        s13=posterior_dict_22['chi_1'][i], s23=posterior_dict_22['chi_2'][i],
        luminosity_distance=posterior_dict_22['luminosity_distance'][i],
        inc=posterior_dict_22['theta_jn'][i], psi=posterior_dict_22['psi'][i],
        phase=posterior_dict_22['phase'][i],
        geocent_time=posterior_dict_22['geocent_time'][i],
        ra=posterior_dict_22['ra'][i], dec=posterior_dict_22['dec'][i])

    likelihood_hom_parameters = dict(
        total_mass=posterior_dict_hom['mass_1'][i]+posterior_dict_hom['mass_2'][i],
        mass_ratio=posterior_dict_hom['mass_2'][i]/posterior_dict_hom['mass_1'][i],
        s13=posterior_dict_hom['chi_1'][i], s23=posterior_dict_hom['chi_2'][i],
        luminosity_distance=posterior_dict_hom['luminosity_distance'][i],
        inc=posterior_dict_hom['theta_jn'][i], psi=posterior_dict_hom['psi'][i],
        phase=posterior_dict_hom['phase'][i]+np.pi/2.,
        geocent_time=posterior_dict_hom['geocent_time'][i],
        ra=posterior_dict_hom['ra'][i], dec=posterior_dict_hom['dec'][i])

    likelihood_hom.parameters = likelihood_hom_parameters
    likelihood_memory.parameters = likelihood_hom_parameters
    likelihood_imr_phenom.parameters = likelihood_imr_parameters

    logger.info("Ethan 22 log likelihood: " + str(ethan_22_log_likelihood[i]))
    logger.info("Restored 22 log likelihood: " + str(likelihood_imr_phenom.log_likelihood_ratio()))
    logger.info("Ethan HOM log likelihood: " + str(ethan_hom_log_likelihood[i]))
    logger.info("Restored HOM log likelihood: " + str(likelihood_hom.log_likelihood_ratio()))
    logger.info("Memory log likelihood: " + str(likelihood_memory.log_likelihood_ratio()))
    logger.info("")

    # likelihood_with_hom = likelihood_hom.log_likelihood_ratio()
    # weight = np.exp(likelihood_with_hom - likelihoods_22[i])
    #
    # likelihoods_hom.append(likelihood_with_hom)
    # weights.append(weight)
    #
    # print(likelihoods_22[i], likelihood_hom, likelihood_with_hom - likelihoods_22[i])
    # print('evalution {}/{}'.format(i, number_of_samples))

print('test')
