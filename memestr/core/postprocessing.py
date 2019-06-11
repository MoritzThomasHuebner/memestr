from copy import deepcopy

import bilby
import bilby.gw.utils as utils
import gwmemory
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import minimize
from collections import namedtuple

import json

from memestr.core.waveforms.surrogate import *
from memestr.core.waveforms.phenom import *

logger = bilby.core.utils.logger
gamma_lmlm = gwmemory.angles.load_gamma()
roll_off = 0.2


class PostprocessingResult(object):

    def __init__(self, outdir, maximum_overlaps=None, memory_log_bf=None, memory_weights=None,
                 hom_log_bf=None, hom_weights=None):
        self.maximum_overlaps = maximum_overlaps
        self.memory_log_bf = memory_log_bf
        self.memory_weights = memory_weights
        self.hom_log_bf = hom_log_bf
        self.hom_weights = hom_weights
        self.outdir = outdir

    def to_json(self):
        with open(self.outdir + 'pp_result.json', 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_json(cls, outdir):
        with open(outdir + 'pp_result.json', 'r') as f:
            data = json.load(f)
        return cls(outdir=outdir, maximum_overlaps=data['maximum_overlaps'],
                   memory_log_bf=data['memory_log_bf'], memory_weights=data['memory_weights'],
                   hom_log_bf=data['hom_log_bf'], hom_weights=data['hom_weights'])

    @property
    def effective_samples(self):
        return np.sum(self.hom_weights) ** 2 / np.sum(np.array(self.hom_weights) ** 2)


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


def calculate_overlaps_optimizable(new_params, *args):
    time_shift = new_params[0]
    phase = new_params[1]
    full_wf, memory_generator, inc, frequency_array, power_spectral_density, alpha = args
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.time_array = memory_generator.times

    waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, inc, phase)
    waveform_fd, _ = convert_to_frequency_domain(memory_generator=memory_generator, series=series,
                                                 waveform=waveform, alpha=alpha, time_shift=time_shift)
    return -overlap_function(a=full_wf, b=waveform_fd, frequency=frequency_array,
                             psd=power_spectral_density)


def get_time_and_phase_shift(parameters, ifo, verbose=False, **kwargs):
    time_limit = parameters['total_mass'] * 0.00030
    if 's13' not in parameters.keys():
        parameters['s13'] = parameters['chi_1']
        parameters['s23'] = parameters['chi_2']
        parameters['inc'] = parameters['theta_jn']

    duration = kwargs.get('duration', 16)
    sampling_frequency = kwargs.get('sampling_frequency', 2048)
    minimum_frequency = kwargs.get('minimum_frequency', 10)
    recovery_wg = bilby.gw.waveform_generator. \
        WaveformGenerator(frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
                          duration=duration, sampling_frequency=sampling_frequency,
                          waveform_arguments=dict(alpha=0.1))
    full_wf = recovery_wg.frequency_domain_strain(parameters)

    memory_generator = gwmemory.waveforms.HybridSurrogate(q=parameters['mass_ratio'],
                                                          total_mass=parameters['total_mass'],
                                                          spin_1=parameters['s13'],
                                                          spin_2=parameters['s23'],
                                                          times=recovery_wg.time_array,
                                                          distance=parameters['luminosity_distance'],
                                                          minimum_frequency=minimum_frequency,
                                                          sampling_frequency=sampling_frequency,
                                                          units='mks',
                                                          )

    logger.info(memory_generator.reference_frequency)

    maximum_overlap = 0.
    time_shift = 0.
    new_phase = 0.
    iterations = 0.
    alpha = 0.1
    args = (full_wf, memory_generator, parameters['inc'],
            recovery_wg.frequency_array, ifo.power_spectral_density, alpha)

    time_limit_start = time_limit
    for threshold in [0.99, 0.95, 0.90, 0.80, 0.60]:
        counter = 0
        time_limit = time_limit_start * threshold
        while maximum_overlap < threshold and counter < 8:
            if counter == 0:
                init_guess_time = 0.
                init_guess_phase = parameters['phase']
                x0 = np.array([init_guess_time, init_guess_phase])
                bounds = [(-time_limit, 0), (parameters['phase'] - np.pi / 2, parameters['phase'] + np.pi / 2)]
            else:
                init_guess_time = -np.random.random() * time_limit
                init_guess_phase = (np.random.random() - 0.5) * np.pi + parameters['phase']
                x0 = np.array([init_guess_time, init_guess_phase])
                bounds = [(-time_limit, 0), (parameters['phase'] - np.pi / 2, parameters['phase'] + np.pi / 2)]
            res = minimize(calculate_overlaps_optimizable, x0=x0, args=args, bounds=bounds, tol=0.00001)

            if -res.fun < maximum_overlap:
                counter += 1
                continue
            maximum_overlap = -res.fun
            time_shift, new_phase = res.x[0], res.x[1]
            new_phase %= 2 * np.pi
            iterations = res.nit
            counter += 1
        if maximum_overlap > threshold:
            break
    # test_gw = bilby.gw.WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
    #                                      start_time=0, duration=16, sampling_frequency=2048,
    #                                      waveform_arguments=dict(time_shift=time_shift, minimum_frequency=20, alpha=alpha))
    # parameters['phase'] = new_phase
    # test_waveform = test_gw.time_domain_strain(parameters)
    # test_waveform_fd = test_gw.frequency_domain_strain(parameters)

    # plt.plot(recovery_wg.time_array, test_waveform['plus'])
    # plt.plot(recovery_wg.time_array, recovery_wg.time_domain_strain()['plus'])
    # plt.show()
    # plt.clf()

    # series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0, sampling_frequency=2048, duration=16)
    # waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, parameters['inc'], new_phase)
    # waveform_fd, _ = convert_to_frequency_domain(memory_generator=memory_generator, series=series,
    #                                              waveform=waveform, alpha=alpha, time_shift=time_shift)
    #
    # print(overlap_function(waveform_fd, test_waveform_fd, recovery_wg.frequency_array, ifo.power_spectral_density))
    # print(overlap_function(waveform_fd, full_wf, recovery_wg.frequency_array, ifo.power_spectral_density))
    # print(overlap_function(test_waveform_fd, full_wf, recovery_wg.frequency_array, ifo.power_spectral_density))

    # plt.loglog()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.abs(waveform_fd['plus']))
    # plt.plot(recovery_wg.frequency_array, np.abs(test_waveform_fd['plus']))
    # plt.show()
    # plt.clf()

    # plt.loglog()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.abs(full_wf['cross']))
    # plt.plot(recovery_wg.frequency_array, np.abs(test_waveform_fd['cross']))
    # plt.show()
    # plt.clf()
    #
    # plt.semilogx()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.angle(full_wf['plus']))
    # plt.plot(recovery_wg.frequency_array, np.angle(test_waveform_fd['plus']))
    # plt.show()
    # plt.clf()
    #
    # plt.semilogx()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.angle(full_wf['cross']))
    # plt.plot(recovery_wg.frequency_array, np.angle(test_waveform_fd['cross']))
    # plt.show()
    # plt.clf()
    #
    if verbose:
        logger.info("Maximum overlap: " + str(maximum_overlap))
        logger.info("Iterations " + str(iterations))
        logger.info("Time shift:" + str(time_shift))
        logger.info("Maximum time shift:" + str(-time_limit))
        logger.info("New Phase:" + str(new_phase))
        logger.info("Counter:" + str(counter))
        logger.info("Threshold:" + str(threshold))

    return time_shift, new_phase, maximum_overlap


def calculate_overlaps_inverted(new_params, *args):
    time_shift = new_params[0]
    phase = new_params[1]
    memory_wf, waveform_generator, inc, frequency_array, power_spectral_density, alpha = args
    waveform_generator.parameters['phase'] = phase
    matching_wf = waveform_generator.frequency_domain_strain()
    matching_wf = apply_time_shift_frequency_domain(waveform=matching_wf, frequency_array=waveform_generator.frequency_array,
                                                    duration=waveform_generator.duration, shift=time_shift)
    return -overlap_function(a=memory_wf, b=matching_wf, frequency=frequency_array,
                             psd=power_spectral_density)


def get_time_and_phase_shift_inverted(parameters, ifo, verbose=False, **kwargs):
    time_limit = parameters['total_mass'] * 0.00030
    if 's13' not in parameters.keys():
        parameters['s13'] = parameters['chi_1']
        parameters['s23'] = parameters['chi_2']
        parameters['inc'] = parameters['theta_jn']

    duration = kwargs.get('duration', 16)
    sampling_frequency = kwargs.get('sampling_frequency', 2048)
    injected_wg = bilby.gw.waveform_generator. \
        WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
                          duration=duration, sampling_frequency=sampling_frequency,
                          waveform_arguments=dict(alpha=0.1))
    recovery_wg = bilby.gw.waveform_generator. \
        WaveformGenerator(frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
                          duration=duration, sampling_frequency=sampling_frequency,
                          waveform_arguments=dict(alpha=0.1))

    memory_wf = injected_wg.frequency_domain_strain(parameters)

    maximum_overlap = 0.
    time_shift = 0.
    new_phase = 0.
    iterations = 0.
    alpha = 0.1
    args = (memory_wf, recovery_wg, parameters['inc'],
            recovery_wg.frequency_array, ifo.power_spectral_density, alpha)

    time_limit_start = time_limit
    for threshold in [0.99, 0.95, 0.90, 0.80, 0.60]:
        counter = 0
        time_limit = time_limit_start * threshold
        while maximum_overlap < threshold and counter < 8:
            if counter == 0:
                init_guess_time = 0.
                init_guess_phase = parameters['phase']
                x0 = np.array([init_guess_time, init_guess_phase])
                bounds = [(0, time_limit), (parameters['phase'] - np.pi / 2, parameters['phase'] + np.pi / 2)]
            else:
                init_guess_time = -np.random.random() * time_limit
                init_guess_phase = (np.random.random() - 0.5) * np.pi + parameters['phase']
                x0 = np.array([init_guess_time, init_guess_phase])
                bounds = [(0, time_limit), (parameters['phase'] - np.pi / 2, parameters['phase'] + np.pi / 2)]
            res = minimize(calculate_overlaps_inverted, x0=x0, args=args, bounds=bounds, tol=0.00001)

            if -res.fun < maximum_overlap:
                counter += 1
                continue
            maximum_overlap = -res.fun
            time_shift, new_phase = res.x[0], res.x[1]
            new_phase %= 2 * np.pi
            iterations = res.nit
            counter += 1
        if maximum_overlap > threshold:
            break

    if verbose:
        logger.info("Maximum overlap: " + str(maximum_overlap))
        logger.info("Iterations " + str(iterations))
        logger.info("Time shift:" + str(time_shift))
        logger.info("Maximum time shift:" + str(-time_limit))
        logger.info("New Phase:" + str(new_phase))
        logger.info("Counter:" + str(counter))
        logger.info("Threshold:" + str(threshold))

    return time_shift, new_phase, maximum_overlap


def adjust_phase_and_geocent_time_complete_posterior_proper(result, ifo, verbose=False, **kwargs):
    new_result = deepcopy(result)
    maximum_overlaps = []
    time_shifts = []
    for index in range(len(result.posterior)):
        parameters = result.posterior.iloc[index].to_dict()
        time_shift, new_phase, maximum_overlap = \
            get_time_and_phase_shift(parameters, ifo, verbose=verbose, **kwargs)
        new_phase %= 2*np.pi
        maximum_overlaps.append(maximum_overlap)
        time_shifts.append(time_shift)
        new_result.posterior.geocent_time.iloc[index] += time_shift
        new_result.posterior.phase.iloc[index] = new_phase
        logger.info(("{:0.2f}".format(index / len(result.posterior) * 100) + "%"))
    return new_result, time_shifts, maximum_overlaps


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


def calculate_log_weights(new_likelihood, new_result, reference_likelihood, reference_result=None):
    if reference_result is None:
        reference_result = deepcopy(new_result)
    log_weights = []

    for i in range(len(new_result.posterior)):
        if i % 100 == 0:
            logger.info("{:0.2f}".format(i / len(new_result.posterior) * 100) + "%")
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                          'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
            new_likelihood.parameters[parameter] = new_result.posterior.iloc[i][parameter]
            reference_likelihood.parameters[parameter] = reference_result.posterior.iloc[i][parameter]

        reweighted_likelihood = new_likelihood.log_likelihood_ratio()
        original_likelihood = reference_likelihood.log_likelihood_ratio()
        weight = reweighted_likelihood - original_likelihood
        log_weights.append(weight)

    return log_weights


def reweigh_log_evidence_by_weights(log_evidence, log_weights):
    return log_evidence + logsumexp(log_weights) - np.log(len(log_weights))


def reweigh_by_likelihood(new_likelihood, new_result, reference_likelihood, reference_result=None):
    try:
        log_weights = calculate_log_weights(new_likelihood=new_likelihood,
                                            new_result=new_result,
                                            reference_likelihood=reference_likelihood,
                                            reference_result=reference_result)
        reweighed_log_bf = logsumexp(log_weights) - np.log(len(log_weights))
    except AttributeError as e:
        logger.warning(e)
        log_weights = np.nan
        reweighed_log_bf = np.nan
    return reweighed_log_bf, log_weights

