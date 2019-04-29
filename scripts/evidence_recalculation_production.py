from scipy.misc import logsumexp
import numpy as np
import bilby as bb
from memestr.core.parameters import AllSettings
from memestr.core.waveforms import time_domain_nr_hyb_sur_waveform_with_memory_wrapped, \
    frequency_domain_IMRPhenomD_waveform_without_memory

import sys
from copy import deepcopy
import pandas as pd

run_id = sys.argv[1]
outdir = run_id + '_reweighing_result'
bb.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

logger = bb.core.utils.logger


def reweigh_evidences(data_dir, alpha=0.1, **kwargs):
    settings = AllSettings.from_defaults_with_some_specified_kwargs(alpha=alpha, **kwargs)
    result = _load_result(data_dir, 'IMR_mem_inj_non_mem_rec_result.json')
    interferometers = bb.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(run_id) + '_H1L1V1.h5')
    for ifo in interferometers:
        logger.info(ifo.name + ' matched_filter_snr: ' + str(ifo.meta_data['matched_filter_SNR']))
    parameters = result.injection_parameters
    settings.injection_parameters.__dict__ = parameters
    priors = result.priors

    settings.detector_settings.zero_noise = False
    waveform_generator_memory = bb.gw.WaveformGenerator(
        time_domain_source_model=time_domain_nr_hyb_sur_waveform_with_memory_wrapped,
        parameters=settings.injection_parameters.__dict__,
        waveform_arguments=settings.waveform_arguments.__dict__,
        **settings.waveform_data.__dict__)
    waveform_generator_no_memory = bb.gw.WaveformGenerator(
        frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
        parameters=settings.injection_parameters.__dict__,
        waveform_arguments=settings.waveform_arguments.__dict__,
        **settings.waveform_data.__dict__)

    likelihood = bb.gw.likelihood \
        .GravitationalWaveTransient(interferometers=interferometers,
                                    waveform_generator=waveform_generator_memory,
                                    priors=priors,
                                    distance_marginalization=True,
                                    time_marginalization=False)

    settings.injection_parameters.__dict__ = deepcopy(parameters)
    waveform_generator_memory.parameters = deepcopy(parameters)

    likelihood.interferometers = interferometers
    likelihood.parameters = deepcopy(parameters)

    evidence_non_memory = likelihood.log_likelihood_ratio()

    likelihood.waveform_generator = waveform_generator_memory
    evidence_memory = likelihood.log_likelihood_ratio()
    logger.info("Injected value log BF: \t" + str(evidence_memory - evidence_non_memory))
    injection_bf = evidence_memory - evidence_non_memory

    likelihood.waveform_generator = waveform_generator_no_memory
    reweighed_log_bf_non_mem_to_self, _ = _reweigh(likelihood, result, waveform_generator_no_memory)
    logger.info("Reweighed no memory to self: \t" + str(reweighed_log_bf_non_mem_to_self))

    likelihood.waveform_generator = waveform_generator_memory
    reweighed_log_bf_mem_inj_to_mem, weights = _reweigh(likelihood, result, waveform_generator_memory)
    logger.info("Reweighed memory inj to memory log BF: \t" + str(reweighed_log_bf_mem_inj_to_mem +
                                                                  reweighed_log_bf_non_mem_to_self))

    res = pd.DataFrame.from_dict(dict(injection_bfs=injection_bf,
                                      reweighing_to_memory_bfs_mem_inj=reweighed_log_bf_mem_inj_to_mem))
    res.to_json(data_dir + '/combined.json')

    # for params in res.posterior:
    #     phase_grid = np.linspace(priors['phase'].minimum, priors['phase'].maximum, 60)
    #     dphi = (phase_grid[1] - phase_grid[0]) / len(phase_grid)
    #     time_grid = np.linspace(priors['geocent_time'].minimum, priors['geocent_time'].maximum, 120)
    #     dt_g = (time_grid[1] - time_grid[0]) / len(time_grid)
    #     phase_grid, time_grid = np.meshgrid(phase_grid, time_grid)
    #     phase_grid = phase_grid.flatten()
    #     time_grid = time_grid.flatten()
    #
    #     matching_wf = frequency_domain_IMRPhenomD_waveform_without_memory(**params)
    #     params_without_phase_and_time = deepcopy(params)
    #     del params_without_phase_and_time['geocent_time']
    #     del params_without_phase_and_time['phase']
    #     full_wf = time_domain_nr_hyb_sur_waveform_with_memory_wrapped(phase=phase_grid, geocent_time=time_grid,
    #                                                                   **params_without_phase_and_time)
    #
    #     overlap = _overlap(matching_wf, full_wf, interferometers[0].strain_data.frequency_array, interferometers[0])
    #
    #     max_nO = np.argmax(overlap)
    #     time_shift = time_grid[max_nO]
    #     phase_new = phase_grid[max_nO]


def _get_sampling_bf(res_mem_inj_mem_rec, res_mem_inj_non_mem_rec):
    try:
        sampling_bf_mem_inj = res_mem_inj_mem_rec.log_evidence - res_mem_inj_non_mem_rec.log_evidence
    except AttributeError as e:
        logger.warning(e)
        sampling_bf_mem_inj = np.nan
    return sampling_bf_mem_inj


def _load_result(outdir, label):
    try:
        res = bb.result.read_in_result(outdir + '/' + label, extension='json')
    except OSError as e:
        logger.warning(e)
        res = None
    return res


def _reweigh(reweighing_likelihood, result, reweighing_waveform):
    reweighing_likelihood.waveform_generator = reweighing_waveform
    try:
        log_weights = _calculate_log_weights(reweighing_likelihood, result.posterior)
        reweighed_log_bf = _reweigh_log_evidence_by_weights(result.log_evidence, log_weights) - result.log_evidence
    except AttributeError as e:
        logger.warning(e)
        reweighed_log_bf = np.nan
        log_weights = np.nan
    return np.exp(log_weights), reweighed_log_bf


def _reweigh_log_evidence_by_weights(log_evidence, log_weights):
    return log_evidence + logsumexp(log_weights) - np.log(len(log_weights))


def _calculate_log_weights(likelihood, posterior):
    weights = []
    for i in range(len(posterior)):
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'phase',
                          'ra', 'dec', 'psi']:
            likelihood.parameters[parameter] = posterior.iloc[i][parameter]
        reweighed_likelihood = likelihood.log_likelihood()
        original_likelihood = posterior.iloc[i]['log_likelihood']
        weight = reweighed_likelihood - original_likelihood
        weights.append(weight)
    return weights


def _overlap(a, b, frequencies, ifo):
    psd = ifo.power_spectral_density
    psd_interp = psd.power_spectral_density_interpolated(frequencies)
    duration = 1./(frequencies[1] - frequencies[0])

    inner_a = bb.gw.utils.noise_weighted_inner_product(a, a, psd_interp, duration)
    inner_b = _vec_inner_prod(b, b, psd_interp, duration)
    inner_ab = _vec_inner_prod(a, b, psd_interp, duration)

    return np.real(inner_ab/np.sqrt(inner_a*inner_b))


def _vec_inner_prod(aa, bb, power_spectral_density, duration):
    integrand = np.conj(aa) * bb / power_spectral_density
    return np.real(4 / duration * np.sum(integrand, axis=1))

reweigh_evidences(data_dir=str(run_id) + '_IMR_mem_inj_non_mem_rec')
