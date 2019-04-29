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

ifos = bb.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(run_id) + 'H1L1V1.h5')
result = bb.result.read_in_result(str(run_id) + '_IMR_mem_inj_non_mem_rec')
parameters = result.injection_parameters
outdir_mem_inj_non_mem_rec = run_id + '_IMR_mem_inj_non_mem_rec'

logger = bb.utils.logger


def reweigh_evidences(run_id, sampling_frequency=2048, duration=16, alpha=0.1):
    settings = AllSettings.from_defaults_with_some_specified_kwargs(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    alpha=alpha)
    settings.injection_parameters.__dict__ = deepcopy(parameters)
    settings.detector_settings.zero_noise = False
    settings.waveform_data.start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
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

    result = _load_result(outdir_mem_inj_non_mem_rec, 'IMR_mem_inj_non_mem_rec_result.json')
    interferometers = bb.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(run_id) + 'H1L1V1.h5')
    priors = result.priors

    likelihood = bb.gw.likelihood \
        .GravitationalWaveTransient(interferometers=interferometers,
                                    waveform_generator=waveform_generator_memory,
                                    priors=priors,
                                    distance_marginalization=True,
                                    time_marginalization=False)

    settings.injection_parameters.__dict__ = deepcopy(parameters)
    waveform_generator_memory.parameters = deepcopy(parameters)

    likelihood.interferometers = bb.gw.detector.InterferometerList(ifos)
    likelihood.parameters = deepcopy(parameters)

    evidence_non_memory = likelihood.log_likelihood_ratio()

    likelihood.waveform_generator = waveform_generator_memory
    evidence_memory = likelihood.log_likelihood_ratio()
    logger.info("Injected value log BF: \t" + str(evidence_memory - evidence_non_memory))
    injection_bf = evidence_memory - evidence_non_memory

    reweighed_log_bf_non_mem_to_self = -_reweigh(likelihood, result, waveform_generator_no_memory)
    logger.info("Reweighed no memory to self: \t" + str(reweighed_log_bf_non_mem_to_self))

    reweighed_log_bf_mem_inj_to_mem = _reweigh(likelihood, result, waveform_generator_memory)
    logger.info("Reweighed memory inj to memory log BF: \t" + str(reweighed_log_bf_mem_inj_to_mem +
                                                                  reweighed_log_bf_non_mem_to_self))

    logger.info(injection_bf)
    res = pd.DataFrame.from_dict(dict(injection_bfs=injection_bf,
                                      reweighing_to_memory_bfs_mem_inj=reweighed_log_bf_mem_inj_to_mem))
    res.to_json(outdir_mem_inj_non_mem_rec + '/combined.json')


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
    return reweighed_log_bf


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


# Use sampling_frequency == 4096 from 0 to 64 and 2048 after that for existing pop runs
# print_evidences(subdirs=[str(subdir) for subdir in range(int(sys.argv[1]), int(sys.argv[2]))],
#                 sampling_frequency=int(sys.argv[3]))
reweigh_evidences(subdirs=[str(subdir) for subdir in range(0, 8)], interferometers=ifos)
