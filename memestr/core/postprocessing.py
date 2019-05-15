from copy import deepcopy

import numpy as np
import bilby.gw.utils as utils
import matplotlib.pyplot as plt
import bilby
import gwmemory
from scipy.misc import logsumexp

from memestr.core.waveforms import wrap_by_time_shift_continuous
from .waveforms import wrap_at_maximum, apply_window, wrap_by_n_indices, \
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


def calculate_overlaps(full_wf, memory_generator, inc, phases, time_shifts,
                       frequency_array, power_spectral_density, shift, **kwargs):

    times = memory_generator.times
    kwargs['alpha'] = 0.1

    overlaps = np.zeros(len(phases) * len(time_shifts))
    time_shifted_waveform = dict()

    for j in range(len(phases)):
        phase_shifted_waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, inc, phases[j])
        phase_shifted_waveform = apply_window(waveform=phase_shifted_waveform, times=times, kwargs=kwargs)
        phase_shifted_waveform = wrap_by_n_indices(shift=shift, waveform=phase_shifted_waveform)

        for i in range(len(time_shifts)):
            target_index = i * len(phases) + j
            for mode in ['plus', 'cross']:
                time_shifted_waveform[mode] = wrap_by_time_shift_continuous(
                    times=memory_generator.times,
                    waveform=phase_shifted_waveform[mode],
                    time_shift=time_shifts[i])
                time_shifted_waveform[mode], _ = bilby.core.utils.nfft(time_shifted_waveform[mode],
                                                                       memory_generator.sampling_frequency)
            overlaps[target_index] = overlap_function(full_wf, time_shifted_waveform,
                                                      frequency_array, power_spectral_density)
        # print("{:0.2f}".format(j/len(phases)*100) + "%")

    return overlaps


def get_time_and_phase_shift(parameters, ifo, plot=False, verbose=False):
    phase_grid_init = np.linspace(0, np.pi, 30)
    time_grid_init = np.linspace(-0.01, -0.0, 30)

    phase_grid_mesh, time_grid_mesh = np.meshgrid(phase_grid_init, time_grid_init)

    phase_grid = phase_grid_mesh.flatten()
    time_grid = time_grid_mesh.flatten()

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
    wrap_check_wf, shift = wrap_at_maximum(wrap_check_wf, dict())

    full_wf = recovery_wg.frequency_domain_strain(parameters)

    phases = (phase_grid_init + parameters['phase']) % (2 * np.pi)

    overlaps = calculate_overlaps(full_wf=full_wf, memory_generator=memory_generator, inc=parameters['inc'],
                                  phases=phases, time_shifts=time_grid_init, shift=shift,
                                  frequency_array=recovery_wg.frequency_array,
                                  power_spectral_density=ifo.power_spectral_density)

    overlaps = np.nan_to_num(overlaps)
    max_n0 = np.argmax(overlaps)
    time_shift = time_grid[max_n0]
    phase_shift = phase_grid[max_n0]

    rs_overlaps = np.reshape(overlaps, (len(time_grid_init), len(phase_grid_init)))
    if plot:
        _plot_time_shifts(overlaps, phase_grid_init, time_grid_init)
        _plot_2d_overlap(rs_overlaps, time_grid_mesh, phase_grid_mesh)

    if verbose:
        logger.info('Maximum overlap: ' + str(overlaps[max_n0]))
        logger.info("Time shift:" + str(time_shift))
        logger.info("Phase shift:" + str(phase_shift))

    return time_shift, phase_shift


def adjust_phase_and_geocent_time_complete_posterior_quick(result, ifo, index=-1, verbose=True, plot=True):
    parameters = result.posterior.iloc[index].to_dict()
    time_shift, phase_shift = get_time_and_phase_shift(parameters, ifo, verbose=verbose, plot=plot)
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
    for index in range(len(result.posterior)):
        parameters = result.posterior.iloc[index].to_dict()
        time_shift, phase_shift = get_time_and_phase_shift(parameters, ifo, verbose=verbose, plot=plot)
        new_result.posterior.geocent_time.iloc[index] += time_shift
        new_result.posterior.phase.iloc[index] += phase_shift
        if new_result.posterior.phase.iloc[index] < 0:
            new_result.posterior.phase.iloc[index] += 2 * np.pi
        new_result.posterior.phase.iloc[index] %= 2 * np.pi
        logger.info(("{:0.2f}".format(index / len(result.posterior) * 100) + "%"))
    return new_result


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


def calculate_log_weights(likelihood, posterior):
    weights = []
    for i in range(len(posterior)):
        logger.info("{:0.2f}".format(i/len(posterior)*100) + "%")
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                          'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
            likelihood.parameters[parameter] = posterior.iloc[i][parameter]
        reweighed_likelihood = likelihood.log_likelihood()
        original_likelihood = posterior.iloc[i]['log_likelihood']
        weight = reweighed_likelihood - original_likelihood
        weights.append(weight)
        logger.info("Weight: " + str(weight))
    return weights


def reweigh_log_evidence_by_weights(log_evidence, log_weights):
    return log_evidence + logsumexp(log_weights) - np.log(len(log_weights))


def reweigh_by_likelihood(reweighing_likelihood, result):
    try:
        log_weights = calculate_log_weights(reweighing_likelihood, result.posterior)
        reweighed_log_bf = reweigh_log_evidence_by_weights(result.log_evidence, log_weights) - result.log_evidence
    except AttributeError as e:
        logger.warning(e)
        log_weights = np.nan
        reweighed_log_bf = np.nan
    return reweighed_log_bf, log_weights


def reweigh_by_two_likelihoods(posterior, likelihood_memory, likelihood_no_memory):
    weights = []
    for i in range(len(posterior)):
        logger.info("{:0.2f}".format(i/len(posterior)*100) + "%")
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                          'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
            likelihood_memory.parameters[parameter] = posterior.iloc[i][parameter]
            likelihood_no_memory.parameters[parameter] = posterior.iloc[i][parameter]
        weight = likelihood_memory.log_likelihood() - likelihood_no_memory.log_likelihood()
        weights.append(weight)
    return logsumexp(weights)
