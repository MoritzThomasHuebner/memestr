import bilby
import numpy as np
from pandas import Series
from scipy.misc import logsumexp
import memestr
from memestr.core.waveforms import *
from copy import deepcopy
import sys

logger = bilby.core.utils.logger

event_id = sys.argv[1]
number_of_parallel_runs = int(sys.argv[2])
run_id = int(sys.argv[3])
# event_id = 'GW150914'
# number_of_parallel_runs = 32
# run_id = 0

data = np.genfromtxt(event_id + '/time_data.dat')
time_of_event = data[0]
start_time = data[1]
duration = data[2]
minimum_frequency = data[3]
sampling_frequency = data[4]


asd_data_file = np.genfromtxt(event_id + '/pr_psd.dat')
ifo_names = []
if len(asd_data_file[0]) == 3:
    ifo_names = ['H1', 'L1']
elif len(asd_data_file[0]) == 4:
    ifo_names = ['H1', 'L1', 'V1']


ifos = bilby.gw.detector.InterferometerList(ifo_names)

for name, ifo in zip(ifo_names, ifos):
    psd = bilby.gw.detector.psd.PowerSpectralDensity.from_amplitude_spectral_density_file(event_id + '/' + name + '_psd.dat')
    ifo.power_spectral_density = psd
    strain = np.loadtxt(event_id + '/' + name + '_frequency_domain_data.dat')
    strain = strain[:, 1] + 1j*strain[:, 2]
    ifo.set_strain_data_from_frequency_domain_strain(strain, sampling_frequency=sampling_frequency,
                                                     duration=duration, start_time=start_time)
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = sampling_frequency/2.
    ifo.power_spectral_density.psd_array = np.minimum(ifo.power_spectral_density.psd_array, 1)

hom_result_ethan = bilby.result.read_in_result(filename=event_id + '/corrected_result.json')
hom_result_posterior_list = np.array_split(hom_result_ethan.posterior, number_of_parallel_runs, axis=0)
hom_result_ethan.posterior = hom_result_posterior_list[run_id]

base_result = bilby.result.read_in_result(filename=event_id + '/22_pe_result.json')
base_result_posterior_list = np.array_split(base_result.posterior, number_of_parallel_runs, axis=0)
base_result.posterior = base_result_posterior_list[run_id]

ethan_result = np.loadtxt(event_id + '/new_likelihoods.dat')
ethan_22_log_likelihood = ethan_result[:, 0]
ethan_posterior_hom_log_likelihood = ethan_result[:, 1]
ethan_weight_log_likelihood = ethan_result[:, 2]

# log_weights = []
# for i in range(len(hom_result_ethan.posterior)):
#     log_weights.append(ethan_posterior_hom_log_likelihood[i] - base_result.posterior.log_likelihood.iloc[i])
#
# log_bf = logsumexp(log_weights) - np.log(len(log_weights))
# log_ethan = np.log(np.sum(ethan_weight_log_likelihood)) - np.log(len(ethan_weight_log_likelihood))
# logger.info(log_bf)
# logger.info(log_ethan)
# sys.exit(0)

# try:
    # raise OSError
    # time_and_phase_shifted_result = bilby.result.read_in_result(event_id + '/time_and_phase_shifted_'
    #                                                             + str(run_id) + '_result.json')
# except OSError as e:
#     logger.warning(e)
#     time_and_phase_shifted_result, time_shifts, maximum_overlaps = \
#         memestr.core.postprocessing.adjust_phase_and_geocent_time_complete_posterior_proper(base_result, ifos[0], True,
#                                                                                             minimum_frequency=20, duration=duration,
#                                                                                             sampling_frequency=sampling_frequency)
#
#     maximum_overlaps = np.array(maximum_overlaps)
#     np.savetxt(event_id + '/moritz_maximum_overlaps_' + str(run_id) + '.txt', maximum_overlaps)
#
#     time_and_phase_shifted_result.label = 'time_and_phase_shifted_' + str(run_id)
#     time_and_phase_shifted_result.outdir = event_id
#     time_and_phase_shifted_result.save_to_file()
    # time_and_phase_shifted_result.plot_corner()


waveform_generator_imr = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
    start_time=start_time, duration=duration, sampling_frequency=sampling_frequency)

waveform_generator_hom_moritz = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
    duration=duration, sampling_frequency=sampling_frequency, start_time=start_time, waveform_arguments=dict(alpha=0.1))

waveform_generator_hom_ethan = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=gws_nominal,
    duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)

waveform_generator_memory = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
    duration=duration, sampling_frequency=sampling_frequency, start_time=start_time)

likelihood_imr_phenom = bilby.gw.likelihood \
    .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                waveform_generator=waveform_generator_imr)

likelihood_hom_moritz = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_hom_moritz)

likelihood_hom_ethan = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_hom_ethan)

likelihood_memory = bilby.gw.likelihood \
    .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                waveform_generator=waveform_generator_memory)

likelihoods_22 = base_result.posterior['log_likelihood']
posterior_dict_22 = deepcopy(base_result.posterior)
# posterior_dict_hom = deepcopy(time_and_phase_shifted_result.posterior)
posterior_dict_hom = deepcopy(hom_result_ethan.posterior)
number_of_samples = len(likelihoods_22)

likelihoods_hom_moritz = []
likelihoods_hom_ethan = []
likelihoods_memory = []


for i in range(len(posterior_dict_hom)):
    if i % 100 == 0:
        logger.info(("{:0.2f}".format(i / len(posterior_dict_22['total_mass']) * 100) + "%"))
    likelihood_imr_parameters = dict(
        total_mass=posterior_dict_22['total_mass'].iloc[i],
        mass_ratio=posterior_dict_22['mass_ratio'].iloc[i],
        s13=posterior_dict_22['chi_1'].iloc[i],
        s23=posterior_dict_22['chi_2'].iloc[i],
        luminosity_distance=posterior_dict_22['luminosity_distance'].iloc[i],
        inc=posterior_dict_22['theta_jn'].iloc[i],
        psi=posterior_dict_22['psi'].iloc[i],
        phase=posterior_dict_22['phase'].iloc[i],
        geocent_time=posterior_dict_22['geocent_time'].iloc[i],
        ra=posterior_dict_22['ra'].iloc[i],
        dec=posterior_dict_22['dec'].iloc[i])
    likelihood_hom_ethan_parameters = dict(
        mass_1=posterior_dict_hom['mass_1'].iloc[i],
        mass_2=posterior_dict_hom['mass_2'].iloc[i],
        chi_1=posterior_dict_hom['chi_1'].iloc[i],
        chi_2=posterior_dict_hom['chi_2'].iloc[i],
        luminosity_distance=posterior_dict_hom['luminosity_distance'].iloc[i],
        theta_jn=posterior_dict_hom['theta_jn'].iloc[i],
        psi=posterior_dict_hom['psi'].iloc[i],
        phase=posterior_dict_hom['phase'].iloc[i],
        geocent_time=posterior_dict_hom['geocent_time'].iloc[i],
        ra=posterior_dict_hom['ra'].iloc[i],
        dec=posterior_dict_hom['dec'].iloc[i])
    likelihood_hom_moritz_parameters = dict(
        total_mass=posterior_dict_hom['total_mass'].iloc[i],
        mass_ratio=posterior_dict_hom['mass_ratio'].iloc[i],
        s13=posterior_dict_hom['chi_1'].iloc[i],
        s23=posterior_dict_hom['chi_2'].iloc[i],
        luminosity_distance=posterior_dict_hom['luminosity_distance'].iloc[i],
        inc=posterior_dict_hom['theta_jn'].iloc[i],
        psi=posterior_dict_hom['psi'].iloc[i],
        phase=posterior_dict_hom['phase'].iloc[i] + np.pi/2,
        geocent_time=posterior_dict_hom['geocent_time'].iloc[i],
        ra=posterior_dict_hom['ra'].iloc[i],
        dec=posterior_dict_hom['dec'].iloc[i])

    likelihood_hom_moritz.parameters = likelihood_hom_moritz_parameters
    likelihood_hom_ethan.parameters = likelihood_hom_ethan_parameters
    likelihood_memory.parameters = likelihood_hom_moritz_parameters
    likelihood_imr_phenom.parameters = likelihood_imr_parameters

    likelihoods_hom_moritz.append(likelihood_hom_moritz.log_likelihood_ratio())
    likelihoods_hom_ethan.append(likelihood_hom_ethan.log_likelihood_ratio())
    likelihoods_memory.append(likelihood_memory.log_likelihood_ratio())

    logger.info("Ethan 22 log likelihood: " + str(ethan_22_log_likelihood[i]))
    # logger.info("Restored 22 log likelihood: " + str(likelihood_imr_phenom.log_likelihood_ratio()))
    logger.info("Ethan posterior HOM log likelihood: " + str(ethan_posterior_hom_log_likelihood[i]))
    logger.info("Ethan restored HOM log likelihood: " + str(likelihoods_hom_ethan[i]))
    logger.info("Moritz HOM log likelihood: " + str(likelihoods_hom_moritz[i]))
    logger.info("Memory log likelihood: " + str(likelihoods_memory[i]))
    logger.info("")


likelihoods_hom_moritz = np.array(likelihoods_hom_moritz)
likelihoods_memory = np.array(likelihoods_memory)
np.savetxt(event_id + '/moritz_hom_log_likelihoods_' + str(run_id) + '.txt', likelihoods_hom_moritz)
np.savetxt(event_id + '/ethan_hom_log_likelihoods_' + str(run_id) + '.txt', likelihoods_hom_ethan)
np.savetxt(event_id + '/moritz_memory_log_likelihoods_' + str(run_id) + '.txt', likelihoods_memory)

likelihoods_22 = np.array([likelihood_22 for likelihood_22 in likelihoods_22])
hom_weights_moritz = likelihoods_hom_moritz - likelihoods_22
hom_weights_ethan = likelihoods_hom_ethan - likelihoods_22
memory_weights = likelihoods_memory - likelihoods_hom_moritz

hom_log_bf_moritz = logsumexp(hom_weights_moritz) - np.log(len(hom_weights_moritz))
hom_log_bf_ethan = np.log(np.sum(hom_weights_ethan)) - np.log(len(hom_weights_ethan))
hom_log_bf_ethan_posterior = logsumexp(ethan_weight_log_likelihood) - np.log(len(ethan_weight_log_likelihood))
memory_log_bf = logsumexp(memory_weights) - np.log(len(memory_weights))

n_effective = np.sum(hom_weights_moritz) ** 2 / np.sum(np.array(hom_weights_moritz) ** 2)
logger.info("Effective number of samples: " + str(n_effective))
logger.info("HOM log BF Moritz: " + str(hom_log_bf_moritz))
logger.info("HOM log BF Ethan: " + str(hom_log_bf_ethan))
logger.info("HOM log BF Ethan posterior: " + str(hom_log_bf_ethan_posterior))
logger.info("Memory log BF: " + str(memory_log_bf))
logger.info(str(run_id))

# np.savetxt(event_id + '/moritz_hom_log_bf' + str(run_id) + '.txt', hom_log_bf)
# np.savetxt(event_id + '/moritz_memory_log_bf' + str(run_id) + '.txt', memory_log_bf)
