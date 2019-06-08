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



# base_result.plot_corner(outdir=event_id)
# base_result.label = '22_pe_total_mass'
# base_result.outdir = event_id
# base_result.save_to_file()


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

likelihoods_22 = base_result.posterior['log_likelihood']
posterior_dict_22 = deepcopy(base_result.posterior)
# posterior_dict_hom = deepcopy(time_and_phase_shifted_result.posterior)
posterior_dict_hom = deepcopy(hom_result_ethan.posterior)
number_of_samples = len(likelihoods_22)

likelihoods_hom = []
likelihoods_memory = []

ethan_result = np.loadtxt(event_id + '/new_likelihoods.dat')
ethan_22_log_likelihood = ethan_result[:, 0]
ethan_hom_log_likelihood = ethan_result[:, 1]
ethan_weight_log_likelihood = ethan_result[:, 2]

for i in range(len(posterior_dict_hom)):

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
    likelihood_hom_parameters = dict(
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

    likelihood_hom.parameters = likelihood_hom_parameters
    likelihood_memory.parameters = likelihood_hom_parameters
    likelihood_imr_phenom.parameters = likelihood_imr_parameters

    likelihoods_hom.append(likelihood_hom.log_likelihood_ratio())
    likelihoods_memory.append(likelihood_memory.log_likelihood_ratio())

    # logger.info("Ethan 22 log likelihood: " + str(ethan_22_log_likelihood[i]))
    # logger.info("Restored 22 log likelihood: " + str(likelihood_imr_phenom.log_likelihood_ratio()))
    # logger.info("Ethan HOM log likelihood: " + str(ethan_hom_log_likelihood[i]))
    # logger.info("Restored HOM log likelihood: " + str(likelihood_hom.log_likelihood_ratio()))
    # logger.info("Memory log likelihood: " + str(likelihood_memory.log_likelihood_ratio()))
    # logger.info("")


likelihoods_hom = np.array(likelihoods_hom)
np.savetxt(event_id + '/moritz_hom_log_likelihoods_' + str(run_id) + '.txt', likelihoods_hom)
likelihoods_memory = np.array(likelihoods_memory)
np.savetxt(event_id + '/moritz_memory_log_likelihoods_' + str(run_id) + '.txt', likelihoods_memory)

likelihoods_22 = np.array([likelihood_22 for likelihood_22 in likelihoods_22])
hom_weights = likelihoods_hom - likelihoods_22
memory_weights = likelihoods_memory - likelihoods_hom

hom_log_bf = logsumexp(hom_weights) - np.log(len(hom_weights))
memory_log_bf = logsumexp(memory_weights) - np.log(len(memory_weights))

logger.info("HOM log BF: " + str(hom_log_bf))
logger.info("Memory log BF: " + str(memory_log_bf))
logger.info(str(run_id))

# np.savetxt(event_id + '/moritz_hom_log_bf' + str(run_id) + '.txt', hom_log_bf)
# np.savetxt(event_id + '/moritz_memory_log_bf' + str(run_id) + '.txt', memory_log_bf)
