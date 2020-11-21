from copy import deepcopy

import bilby.gw.utils as utils
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize
import multiprocessing

from memestr.waveforms.phenom import *

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


def calculate_overlaps_optimizable(new_params, *args):
    time_shift = new_params[0]
    phase = new_params[1]
    full_wf, memory_generator, inc, frequency_array, power_spectral_density, alpha = args
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.time_array = memory_generator.times

    waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, inc, phase)
    waveform_fd = convert_to_frequency_domain(
        memory_generator=memory_generator, series=series, waveform=waveform, alpha=alpha,
        time_shift=time_shift, inc=inc, phase=phase)
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
        WaveformGenerator(frequency_domain_source_model=kwargs['recovery_model'],
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
    if verbose:
        logger.info("Maximum overlap: " + str(maximum_overlap))
        logger.info("Iterations " + str(iterations))
        logger.info("Time shift:" + str(time_shift))
        logger.info("Maximum time shift:" + str(-time_limit))
        logger.info("New Phase:" + str(new_phase))
        logger.info("Counter:" + str(counter))
        logger.info("Threshold:" + str(threshold))

    return time_shift, new_phase, maximum_overlap


def adjust_phase_and_geocent_time_complete_posterior_parallel(result, n_parallel=2):
    p = multiprocessing.Pool(n_parallel)
    new_result = deepcopy(result)
    posteriors = np.array_split(new_result.posterior, n_parallel, 0)
    new_results = []
    for i in range(n_parallel):
        res = deepcopy(new_result)
        res.posterior = posteriors[i]
        new_results.append(res)
    shifted_results = p.map(adjust_phase_and_geocent_time_default, new_results)
    shifted_combined_posterior = pd.concat([r.posterior for r in shifted_results])
    new_result.posterior = shifted_combined_posterior
    return new_result


def adjust_phase_and_geocent_time_default(result):
    ifo = bilby.gw.detector.get_empty_interferometer('H1')
    verbose = False
    new_res, max_overlap = adjust_phase_and_geocent_time_complete_posterior_proper(result, ifo, verbose)
    return new_res


def adjust_phase_and_geocent_time_complete_posterior_proper(result, ifo, verbose=False, **kwargs):
    new_result = deepcopy(result)
    maximum_overlaps = []
    for index in range(len(result.posterior)):
        parameters = result.posterior.iloc[index].to_dict()
        time_shift, new_phase, maximum_overlap = \
            get_time_and_phase_shift(parameters, ifo, verbose=verbose, **kwargs)
        new_phase %= 2*np.pi
        maximum_overlaps.append(maximum_overlap)
        new_result.posterior.geocent_time.iloc[index] += time_shift
        new_result.posterior.phase.iloc[index] = new_phase
        # logger.info(("{:0.2f}".format(index / len(result.posterior) * 100) + "%"))
    return new_result, maximum_overlaps


def _plot_2d_overlap(reshaped_overlaps, time_grid_mesh, phase_grid_mesh):
    import matplotlib.pyplot as plt
    plt.contourf(time_grid_mesh, phase_grid_mesh, reshaped_overlaps)
    plt.xlabel('Time shift')
    plt.ylabel('Phase shift')
    plt.colorbar()
    plt.title('Overlap')
    plt.show()
    plt.clf()


def _plot_time_shifts(overlaps, phase_grid_init, time_grid_init):
    rs_overlaps = np.reshape(overlaps, (len(time_grid_init), len(phase_grid_init)))
    import matplotlib.pyplot as plt
    for overlap in rs_overlaps.T:
        plt.plot(time_grid_init, overlap)
        plt.xlabel('Time shift')
        plt.ylabel('Overlap')
    plt.show()
    plt.clf()
    return rs_overlaps


def calculate_log_weights(new_likelihood, result, reference_likelihood, use_stored_likelihood=False):
    log_weights = []

    for i in range(len(result.posterior)):
        new_likelihood.parameters = result.posterior.iloc[i].to_dict()
        reference_likelihood.parameters = result.posterior.iloc[i].to_dict()
        reweighted_likelihood = new_likelihood.log_likelihood_ratio()
        if use_stored_likelihood:
            original_likelihood = result.posterior.log_likelihood.iloc[i]
        else:
            original_likelihood = reference_likelihood.log_likelihood_ratio()

        log_weight = reweighted_likelihood - original_likelihood
        log_weights.append(log_weight)
        if i % 100 == 0:
            logger.info("{:0.2f}".format(i / len(result.posterior) * 100) + "%")
            logger.info("Original Log Likelihood: " + str(original_likelihood))
            logger.info("Reweighted Log Likelihood: " + str(reweighted_likelihood))
            logger.info("Log Weight: " + str(log_weight))

    return log_weights


def reweigh_log_evidence_by_weights(log_evidence, log_weights):
    return log_evidence + logsumexp(log_weights) - np.log(len(log_weights))


def reweigh_by_likelihood(result, new_likelihood, reference_likelihood, use_stored_likelihood=True):
    log_weights = calculate_log_weights(new_likelihood=new_likelihood,
                                        result=result,
                                        reference_likelihood=reference_likelihood,
                                        use_stored_likelihood=use_stored_likelihood)
    reweighted_log_bf = logsumexp(log_weights) - np.log(len(log_weights))
    return reweighted_log_bf, log_weights
