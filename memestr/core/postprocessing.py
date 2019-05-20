from copy import deepcopy
import time

import numpy as np
import bilby.gw.utils as utils
import matplotlib.pyplot as plt
import bilby
import gwmemory
from scipy.misc import logsumexp
from scipy.optimize import minimize

from memestr.core.waveforms import wrap_by_time_shift_continuous, wrap_by_time_shift_continuous_vectorized
from .waveforms import wrap_at_maximum, apply_window, wrap_by_n_indices, apply_window_vectorized, \
    wrap_by_n_indices_vectorized, time_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return, \
    frequency_domain_IMRPhenomD_waveform_without_memory

gamma_lmlm = gwmemory.angles.load_gamma()

logger = bilby.core.utils.logger

roll_off = 0.2


def overlap_function(a, b, frequency, psd):
    psd_interp = psd.power_spectral_density_interpolated(frequency)
    duration = 1. / (frequency[1] - frequency[0])

    inner_a = utils.noise_weighted_inner_product(
        a['plus'], a['plus'], psd_interp, duration)

    inner_a += utils.noise_weighted_inner_product(
        a['cross'], a['cross'], psd_interp, duration)

    inner_b = utils.noise_weighted_inner_product(
        b['plus'], b['plus'], psd_interp, duration)

    inner_b += utils.noise_weighted_inner_product(
        b['cross'], b['cross'], psd_interp, duration)

    inner_ab = utils.noise_weighted_inner_product(
        a['plus'], b['plus'], psd_interp, duration)

    inner_ab += utils.noise_weighted_inner_product(
        a['cross'], b['cross'], psd_interp, duration)
    overlap = inner_ab / np.sqrt(inner_a * inner_b)
    return overlap.real


def nfft_vectorizable(time_domain_strain, sampling_frequency):
    frequency_domain_strain = dict()
    for mode in time_domain_strain:
        frequency_domain_strain[mode] = np.fft.rfft(time_domain_strain[mode])
        frequency_domain_strain[mode] /= sampling_frequency
    return frequency_domain_strain


overlap_function_vectorized = np.vectorize(pyfunc=overlap_function, excluded=['a', 'frequency', 'psd'])
nfft_vectorized = np.vectorize(pyfunc=nfft_vectorizable, excluded=['sampling_frequency'])


# def calculate_overlaps(full_wf, memory_generator, inc, phases, time_shifts,
#                        frequency_array, power_spectral_density, shift, **kwargs):
#     times = memory_generator.times
#     kwargs['alpha'] = 0.1
#     overlaps = np.zeros(len(phases) * len(time_shifts))
#     max_overlap_wf = None
#     time_shifted_waveform_fd = dict()
#     for j in range(len(phases)):
#         phase_shifted_waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, inc, phases[j])
#         phase_shifted_waveform = apply_window(waveform=phase_shifted_waveform, times=times, kwargs=kwargs)
#         phase_shifted_waveform = wrap_by_n_indices(shift=shift, waveform=phase_shifted_waveform)
#
#         for i in range(len(time_shifts)):
#             target_index = i * len(phases) + j
#             time_shifted_waveform = wrap_by_time_shift_continuous(
#                 times=memory_generator.times,
#                 waveform=phase_shifted_waveform,
#                 time_shift=time_shifts[i])
#
#             for mode in ['plus', 'cross']:
#                 time_shifted_waveform_fd[mode], _ = bilby.core.utils.nfft(time_shifted_waveform[mode],
#                                                                           memory_generator.sampling_frequency)
#                 indexes = np.where(frequency_array < 20)
#                 time_shifted_waveform_fd[mode][indexes] = 0
#             overlaps[target_index] = overlap_function(full_wf, time_shifted_waveform_fd,
#                                                       frequency_array, power_spectral_density)
#             if overlaps[target_index] == np.max(overlaps):
#                 max_overlap_wf = time_shifted_waveform
#     return overlaps, max_overlap_wf


# def calculate_overlaps_vectorized(full_wf, memory_generator, inc, phases, time_shifts,
#                                   frequency_array, power_spectral_density, shift, **kwargs):
#     times = memory_generator.times
#     kwargs['alpha'] = 0.1
#
#     phase_shifted_waveform = gwmemory.waveforms.combine_modes_vectorized(memory_generator.h_lm, inc, phases)
#     phase_shifted_waveform = apply_window_vectorized(waveform=phase_shifted_waveform, times=times, kwargs=kwargs)
#     phase_shifted_waveform = wrap_by_n_indices_vectorized(shift=shift, waveform=phase_shifted_waveform)
#
#     time_shifted_waveform = np.array([])
#     for ts in time_shifts:
#         time_shifted_waveform = np.append(time_shifted_waveform, wrap_by_time_shift_continuous_vectorized(
#             times=memory_generator.times,
#             waveform=phase_shifted_waveform,
#             time_shift=ts))
#
#     time_shifted_waveform = nfft_vectorized(time_shifted_waveform, memory_generator.sampling_frequency)
#
#     return overlap_function_vectorized(a=full_wf, b=time_shifted_waveform, frequency=frequency_array,
#                                        psd=power_spectral_density)
#

def calculate_overlaps_optimizable(new_params, *args):
    time_shift = new_params[0]
    phase = new_params[1]
    full_wf, memory_generator, inc, frequency_array, power_spectral_density, shift, alpha = args
    times = memory_generator.times
    kwargs = dict(alpha=alpha)

    phase_shifted_waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, inc, phase)
    phase_shifted_waveform = apply_window(waveform=phase_shifted_waveform, times=times, kwargs=kwargs)
    phase_shifted_waveform = wrap_by_n_indices(shift=shift, waveform=phase_shifted_waveform)

    time_shifted_waveform = wrap_by_time_shift_continuous(
            times=memory_generator.times,
            waveform=phase_shifted_waveform,
            time_shift=time_shift)

    time_shifted_waveform_fd = nfft_vectorizable(time_shifted_waveform, memory_generator.sampling_frequency)
    for mode in ['plus', 'cross']:
        time_shifted_waveform_fd[mode], _ = bilby.core.utils.nfft(time_shifted_waveform[mode],
                                                                  memory_generator.sampling_frequency)
        indexes = np.where(frequency_array < 20)
        time_shifted_waveform_fd[mode][indexes] = 0

    return -overlap_function(a=full_wf, b=time_shifted_waveform_fd, frequency=frequency_array,
                             psd=power_spectral_density)


def get_time_and_phase_shift(parameters, ifo, verbose=False):
    time_limit = parameters['total_mass'] * 0.00025

    recovery_wg = bilby.gw.waveform_generator. \
        WaveformGenerator(frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
                          duration=16, sampling_frequency=2048,
                          waveform_arguments=dict(alpha=0.1))

    memory_generator = gwmemory.waveforms.HybridSurrogate(q=parameters['mass_ratio'],
                                                          total_mass=parameters['total_mass'],
                                                          spin_1=parameters['s13'],
                                                          spin_2=parameters['s23'],
                                                          times=recovery_wg.time_array,
                                                          distance=parameters['luminosity_distance'],
                                                          minimum_frequency=10,
                                                          sampling_frequency=2048,
                                                          units='mks',
                                                          )

    wrap_check_wf = gwmemory.waveforms.combine_modes(memory_generator.h_lm, parameters['inc'], parameters['phase'])
    wrap_check_wf, shift = wrap_at_maximum(wrap_check_wf)

    full_wf = recovery_wg.frequency_domain_strain(parameters)

    counter = 0
    maximum_overlap = 0
    time_shift = 0
    phase_shift = 0
    iterations = 0
    args = (full_wf, memory_generator, parameters['inc'],
            recovery_wg.frequency_array, ifo.power_spectral_density, shift, 0.1)

    while maximum_overlap < 0.95:
        init_guess_time = -np.random.random() * time_limit
        init_guess_phase = np.pi*np.random.random()
        x0 = np.array([init_guess_time, init_guess_phase])
        bounds = [(-time_limit, 0.), (0, 2 * np.pi)]
        res = minimize(calculate_overlaps_optimizable, x0=x0, args=args, bounds=bounds,
                       tol=0.00001)
        time_shift, phase_shift = res.x[0], res.x[1]
        maximum_overlap = -res.fun
        iterations = res.nit
        counter += 1
        if counter > 99:
            break

    if verbose:
        logger.info("Maximum overlap: " + str(maximum_overlap))
        logger.info("Iterations " + str(iterations))
        logger.info("Time shift:" + str(time_shift))
        logger.info("Phase shift:" + str(phase_shift))

    return time_shift, phase_shift, shift, maximum_overlap


def adjust_phase_and_geocent_time_complete_posterior_quick(result, ifo, index=-1, verbose=True, plot=True):
    parameters = result.posterior.iloc[index].to_dict()
    time_shift, phase_shift, shift, overlaps = get_time_and_phase_shift(parameters, ifo, verbose=verbose)
    new_result = deepcopy(result)
    for i in range(len(new_result.posterior['geocent_time'])):
        new_result.posterior.geocent_time.iloc[i] += time_shift
        new_result.posterior.phase.iloc[i] += phase_shift
        if new_result.posterior.phase.iloc[i] < 0:
            new_result.posterior.phase.iloc[i] += 2 * np.pi
        new_result.posterior.phase.iloc[i] %= 2 * np.pi
    return new_result


def adjust_phase_and_geocent_time_complete_posterior_proper(result, ifo, verbose=False, plot=False):
    new_result = deepcopy(result)
    maximum_overlaps = []
    shifts = []
    for index in range(len(result.posterior)):
        parameters = result.posterior.iloc[index].to_dict()
        time_shift, phase_shift, shift, maximum_overlap = \
            get_time_and_phase_shift(parameters, ifo, verbose=verbose)
        maximum_overlaps.append(maximum_overlap)
        shifts.append(shift)
        new_result.posterior.geocent_time.iloc[index] += time_shift
        new_result.posterior.phase.iloc[index] += phase_shift
        if new_result.posterior.phase.iloc[index] < 0:
            new_result.posterior.phase.iloc[index] += 2 * np.pi
        new_result.posterior.phase.iloc[index] %= 2 * np.pi
        logger.info(("{:0.2f}".format(index / len(result.posterior) * 100) + "%"))
    return new_result, shifts, maximum_overlaps


def _plot_2d_overlap(reshaped_overlaps, time_grid_mesh, phase_grid_mesh):
    plt.contourf(time_grid_mesh, phase_grid_mesh, reshaped_overlaps)
    plt.xlabel('Time shift')
    plt.ylabel('Phase shift')
    plt.colorbar()
    plt.title('Overlap')
    plt.show()
    plt.clf()


def _plot_time_shifts(overlaps, phase_grid_init, time_grid_init):
    rs_overlaps = np.reshape(overlaps, (len(time_grid_init), len(phase_grid_init)))
    for overlap in rs_overlaps.T:
        plt.plot(time_grid_init, overlap)
        plt.xlabel('Time shift')
        plt.ylabel('Overlap')
    plt.show()
    plt.clf()
    return rs_overlaps


def calculate_log_weights(likelihood, posterior, **kwargs):
    weights = []
    shifts = kwargs.get('shifts')

    for i in range(len(posterior)):
        if i % 100 == 0:
            logger.info("{:0.2f}".format(i / len(posterior) * 100) + "%")
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                          'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
            likelihood.parameters[parameter] = posterior.iloc[i][parameter]
            if shifts is not None:
                likelihood.waveform_generator.waveform_arguments['shift'] = shifts[i]
        reweighted_likelihood = likelihood.log_likelihood_ratio()
        original_likelihood = posterior.iloc[i]['log_likelihood']
        weight = reweighted_likelihood - original_likelihood
        weights.append(weight)
        logger.info("Original Likelihood: " + str(original_likelihood))
        logger.info("Reweighted Likelihood: " + str(reweighted_likelihood))
        logger.info("Weight: " + str(weight))
    return weights


def reweigh_log_evidence_by_weights(log_evidence, log_weights):
    return log_evidence + logsumexp(log_weights) - np.log(len(log_weights))


def reweigh_by_likelihood(reweighing_likelihood, result, **kwargs):
    try:
        log_weights = calculate_log_weights(reweighing_likelihood, result.posterior, **kwargs)
        reweighed_log_bf = reweigh_log_evidence_by_weights(result.log_evidence, log_weights) - result.log_evidence
    except AttributeError as e:
        logger.warning(e)
        log_weights = np.nan
        reweighed_log_bf = np.nan
    return reweighed_log_bf, log_weights


def reweigh_by_two_likelihoods(posterior, likelihood_memory, likelihood_no_memory, **kwargs):
    weights = []
    shifts = kwargs.get('shifts')

    for i in range(len(posterior)):
        logger.info("{:0.2f}".format(i / len(posterior) * 100) + "%")
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                          'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
            likelihood_memory.parameters[parameter] = posterior.iloc[i][parameter]
            likelihood_no_memory.parameters[parameter] = posterior.iloc[i][parameter]
            if shifts is not None:
                likelihood_memory.waveform_generator.waveform_arguments['shift'] = shifts[i]
                likelihood_no_memory.waveform_generator.waveform_arguments['shift'] = shifts[i]

            weight_2 = likelihood_no_memory.log_likelihood_ratio()
            weight_1 = likelihood_memory.log_likelihood_ratio()
            weight = weight_1 - weight_2
        weights.append(weight)
    return logsumexp(weights)
