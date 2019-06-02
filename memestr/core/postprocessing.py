from copy import deepcopy

import bilby
import bilby.gw.utils as utils
import gwmemory
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import minimize

from .waveforms.phenom import frequency_domain_IMRPhenomD_waveform_without_memory
from .waveforms.surrogate import convert_to_frequency_domain
from .waveforms.utils import nfft

logger = bilby.core.utils.logger
gamma_lmlm = gwmemory.angles.load_gamma()
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


overlap_function_vectorized = np.vectorize(pyfunc=overlap_function, excluded=['a', 'frequency', 'psd'])
nfft_vectorized = np.vectorize(pyfunc=nfft, excluded=['sampling_frequency'])


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


def get_time_and_phase_shift(parameters, ifo, verbose=False):
    time_limit = parameters['total_mass'] * 0.00080 * 2.0

    recovery_wg = bilby.gw.waveform_generator. \
        WaveformGenerator(frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
                          duration=16, sampling_frequency=2048,
                          waveform_arguments=dict(alpha=0.1))

    try:
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
    except ValueError as e:
        logger.warning(e)
        memory_generator = gwmemory.waveforms.HybridSurrogate(q=parameters['mass_ratio'],
                                                              total_mass=parameters['total_mass'],
                                                              spin_1=parameters['s13'],
                                                              spin_2=parameters['s23'],
                                                              times=recovery_wg.time_array,
                                                              distance=parameters['luminosity_distance'],
                                                              minimum_frequency=20,
                                                              sampling_frequency=2048,
                                                              units='mks',
                                                              )
    logger.info(memory_generator.reference_frequency)
    full_wf = recovery_wg.frequency_domain_strain(parameters)

    counter = 0.
    maximum_overlap = 0.
    time_shift = 0.
    new_phase = 0.
    iterations = 0.
    alpha = 0.1
    args = (full_wf, memory_generator, parameters['inc'],
            recovery_wg.frequency_array, ifo.power_spectral_density, alpha)
    init_guess_time = -0.5 * time_limit
    init_guess_phase = parameters['phase']
    x0 = np.array([init_guess_time, init_guess_phase])
    bounds = [(-time_limit, 0), (parameters['phase']-np.pi/2, parameters['phase']+np.pi/2)]

    while maximum_overlap < 0.98:
        res = minimize(calculate_overlaps_optimizable, x0=x0, args=args, bounds=bounds,
                       tol=0.00001)
        time_shift, new_phase = res.x[0], res.x[1]
        new_phase %= 2*np.pi
        maximum_overlap = -res.fun
        iterations = res.nit
        init_guess_time = -np.random.random() * time_limit
        init_guess_phase = np.pi*(np.random.random() - 0.5) + parameters['phase']
        x0 = np.array([init_guess_time, init_guess_phase])
        counter += 1
        if counter > 20:
            break
    # test_waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, parameters['inc'], phase_shift)
    # test_waveform = apply_window(waveform=test_waveform, times=recovery_wg.time_array, kwargs=dict(alpha=alpha))
    # test_waveform_fd = dict()
    #
    # for mode in ['plus', 'cross']:
    #     test_waveform_fd[mode], frequency_array = bilby.core.utils.nfft(test_waveform[mode], memory_generator.sampling_frequency)
    #     indexes = np.where(frequency_array < 20)
    #     test_waveform_fd[mode][indexes] = 0
    # duration = memory_generator.times[-1] - memory_generator.times[0]
    # delta_t = memory_generator.times[1] - memory_generator.times[0]
    # absolute_shift = delta_t * shift
    # test_waveform_fd['plus'] = test_waveform_fd['plus']*np.exp(-2j * np.pi * (duration + time_shift + absolute_shift) * frequency_array)
    # test_waveform_fd['cross'] = test_waveform_fd['cross']*np.exp(-2j * np.pi * (duration + time_shift + absolute_shift) * frequency_array)
    #
    # test_waveform['plus'] = bilby.core.utils.infft(test_waveform_fd['plus'], sampling_frequency=2048)
    # test_waveform['cross'] = bilby.core.utils.infft(test_waveform_fd['cross'], sampling_frequency=2048)
    # plt.plot(test_waveform['plus'])
    # plt.show()
    # plt.clf()
    #
    # print(overlap_function(test_waveform_fd, full_wf, recovery_wg.frequency_array, ifo.power_spectral_density))
    #
    # plt.loglog()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.abs(full_wf['plus']))
    # plt.plot(recovery_wg.frequency_array, np.abs(test_waveform_fd['plus']))
    # plt.show()
    # plt.clf()
    #
    # plt.loglog()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.abs(full_wf['cross']))
    # plt.plot(recovery_wg.frequency_array, np.abs(test_waveform_fd['cross']))
    # plt.show()
    # plt.clf()
    #
    # psd_interp = ifo.power_spectral_density.power_spectral_density_interpolated(recovery_wg.frequency_array)
    #
    # plt.loglog()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.abs(full_wf['plus'] - test_waveform_fd['plus'])/np.sqrt(psd_interp))
    # plt.show()
    # plt.clf()
    #
    # plt.loglog()
    # plt.xlim(20, 1024)
    # plt.plot(recovery_wg.frequency_array, np.abs(full_wf['cross'] - test_waveform_fd['cross'])/np.sqrt(psd_interp))
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
        logger.info("New Phase:" + str(new_phase))
        logger.info("Counter:" + str(counter))

    return time_shift, new_phase, maximum_overlap


def adjust_phase_and_geocent_time_complete_posterior_proper(result, ifo, verbose=False):
    new_result = deepcopy(result)
    maximum_overlaps = []
    time_shifts = []
    for index in range(len(result.posterior)):
        parameters = result.posterior.iloc[index].to_dict()
        time_shift, new_phase, maximum_overlap = \
            get_time_and_phase_shift(parameters, ifo, verbose=verbose)
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


def calculate_log_weights(likelihood, result, **kwargs):
    log_weights = []
    # shifts = kwargs.get('shifts')
    test_original_likelihood = kwargs.get('test_original_likelihood')
    test_original_result = kwargs.get('test_original_result')

    for i in range(len(result.posterior)):
        if i % 100 == 0:
            logger.info("{:0.2f}".format(i / len(result.posterior) * 100) + "%")
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                          'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
            likelihood.parameters[parameter] = result.posterior.iloc[i][parameter]
            # if shifts is not None:
            #     pass # Do this to check the overlap
                # likelihood.waveform_generator.waveform_arguments['time_shift'] = shifts[i]
                # likelihood.waveform_generator.waveform_arguments['time_shift'] = \
                #      result.posterior.iloc[i]['geocent_time'] - test_original_result.posterior.iloc[i]['geocent_time']
            if test_original_likelihood is not None:
                test_original_likelihood.parameters[parameter] = test_original_result.posterior.iloc[i][parameter]

        reweighted_likelihood = likelihood.log_likelihood_ratio()
        original_likelihood = test_original_likelihood.log_likelihood_ratio()
        # original_likelihood = result.posterior.iloc[i]['log_likelihood']
        weight = reweighted_likelihood - original_likelihood
        log_weights.append(weight)
        # logger.info("Parameters Likelihood: " + str(likelihood.parameters))
        # if test_original_likelihood is not None:
        #     logger.info("Original Parameters Likelihood: " + str(test_original_likelihood.parameters))

        logger.info("Original Likelihood: " + str(original_likelihood))
        # if test_original_likelihood is not None:
            # logger.info("Original Likelihood Test: " + str(test_original_likelihood.log_likelihood_ratio()))
        logger.info("Reweighted Likelihood: " + str(reweighted_likelihood))
        logger.info("Log weight: " + str(weight))
        # full_wf = test_original_likelihood.waveform_generator.frequency_domain_strain(test_original_likelihood.parameters)
        # matching_wf = likelihood.waveform_generator.frequency_domain_strain(likelihood.parameters)
        # overlap = overlap_function(full_wf, matching_wf, likelihood.waveform_generator.frequency_array,
        #                            likelihood.interferometers[0].power_spectral_density)
        # logger.info("Test Overlap: " + str(overlap))
        # logger.info("")


        # import matplotlib.pyplot as plt
        # plt.plot(test_original_likelihood.waveform_generator.time_array,
        #          test_original_likelihood.waveform_generator.time_domain_strain(test_original_likelihood.parameters)['plus'])
        # plt.plot(test_original_likelihood.waveform_generator.time_array,
        #          likelihood.waveform_generator.time_domain_strain(likelihood.parameters)['plus'])
        # plt.xlim(6, 7.1)
        # plt.show()
        # plt.clf()
        #
        # plt.plot(test_original_likelihood.waveform_generator.time_array,
        #          test_original_likelihood.waveform_generator.time_domain_strain(test_original_likelihood.parameters)['cross'])
        # plt.plot(test_original_likelihood.waveform_generator.time_array,
        #          likelihood.waveform_generator.time_domain_strain(likelihood.parameters)['cross'])
        # plt.xlim(6, 7.1)
        # plt.show()
        # plt.clf()
        #
        # plt.plot(test_original_likelihood.waveform_generator.frequency_array,
        #          np.abs(full_wf['plus']))
        # plt.plot(test_original_likelihood.waveform_generator.frequency_array,
        #          np.abs(matching_wf['plus']))
        # plt.loglog()
        # plt.xlim(20, 1024)
        # plt.show()
        # plt.clf()

        # plt.plot(test_original_likelihood.waveform_generator.frequency_array,
        #          np.abs(full_wf['cross']))
        # plt.plot(test_original_likelihood.waveform_generator.frequency_array,
        #          np.abs(matching_wf['cross']))
        # plt.loglog()
        # plt.xlim(20, 1024)
        # plt.show()
        # plt.clf()

        # plt.semilogx()
        # plt.xlim(20, 1024)
        # plt.plot(test_original_likelihood.waveform_generator.frequency_array, np.angle(full_wf['plus']))
        # plt.plot(test_original_likelihood.waveform_generator.frequency_array, np.angle(matching_wf['plus']))
        # plt.show()
        # plt.clf()

        # plt.semilogx()
        # plt.xlim(20, 1024)
        # plt.plot(test_original_likelihood.waveform_generator.frequency_array, np.angle(full_wf['cross']))
        # plt.plot(test_original_likelihood.waveform_generator.frequency_array, np.angle(matching_wf['cross']))
        # plt.show()
        # plt.clf()
        # import sys
        # sys.exit(0)

    return log_weights


def reweigh_log_evidence_by_weights(log_evidence, log_weights):
    return log_evidence + logsumexp(log_weights) - np.log(len(log_weights))


def reweigh_by_likelihood(reweighing_likelihood, result, **kwargs):
    try:
        log_weights = calculate_log_weights(reweighing_likelihood, result, **kwargs)
        # reweighed_log_bf = reweigh_log_evidence_by_weights(result.log_evidence, log_weights) - result.log_evidence
        reweighed_log_bf = logsumexp(log_weights)
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
    return logsumexp(weights), weights
