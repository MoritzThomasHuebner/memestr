from scipy.misc import logsumexp
import numpy as np
import sys
import bilby as bb
import memestr
from memestr.core.parameters import AllSettings
from memestr.core.waveforms import time_domain_IMRPhenomD_waveform_with_memory, \
    time_domain_IMRPhenomD_waveform_without_memory
import logging
import pandas as pd

outdir = 'evidence_reweighing/parameter_set_0'

outdir_mem_inj_mem_rec = 'evidence_reweighing/parameter_set_0/IMR_mem_inj_mem_rec'
outdir_mem_inj_non_mem_rec = 'evidence_reweighing/parameter_set_0/IMR_mem_inj_non_mem_rec'
outdir_non_mem_inj_mem_rec = 'evidence_reweighing/parameter_set_0/IMR_non_mem_inj_mem_rec'
outdir_non_mem_inj_non_mem_rec = 'evidence_reweighing/parameter_set_0/IMR_non_mem_inj_non_mem_rec'

logger = logging.getLogger('bilby')

injection_bfs = []
sampling_bfs = []
reweighing_to_memory_bfs_mem_inj = []
reweighing_to_memory_bfs_non_mem_inj = []
reweighing_from_memory_bfs_mem_inj = []
reweighing_from_memory_bfs_non_mem_inj = []


def print_evidences(subdirs, sampling_frequency=2048, duration=16, alpha=0.1):
    settings = AllSettings.from_defaults_with_some_specified_kwargs(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    alpha=alpha)
    waveform_generator_memory = bb.gw.WaveformGenerator(
        time_domain_source_model=time_domain_IMRPhenomD_waveform_with_memory,
        parameters=settings.injection_parameters.__dict__,
        waveform_arguments=settings.waveform_arguments.__dict__,
        **settings.waveform_data.__dict__)

    waveform_generator_no_memory = bb.gw.WaveformGenerator(
        time_domain_source_model=time_domain_IMRPhenomD_waveform_without_memory,
        parameters=settings.injection_parameters.__dict__,
        waveform_arguments=settings.waveform_arguments.__dict__,
        **settings.waveform_data.__dict__)

    logger.disabled = True
    ifos = [bb.gw.detector.get_interferometer_with_fake_noise_and_injection(
        name=name,
        injection_polarizations=waveform_generator_memory.frequency_domain_strain(),
        injection_parameters=settings.injection_parameters.__dict__,
        outdir=outdir,
        zero_noise=True,
        plot=False,
        **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors]
    logger.disabled = False
    priors = dict(luminosity_distance=bb.gw.prior.UniformComovingVolume(name='luminosity_distance',
                                                                        minimum=10,
                                                                        maximum=2000))
    likelihood = bb.gw.likelihood \
        .GravitationalWaveTransient(interferometers=ifos,
                                    waveform_generator=waveform_generator_memory,
                                    priors=priors,
                                    distance_marginalization=True)

    for subdir in subdirs:

        res_mem_inj_mem_rec = _load_result(outdir_mem_inj_mem_rec, subdir, 'IMR_mem_inj_mem_rec_result.json')
        res_mem_inj_non_mem_rec = _load_result(outdir_mem_inj_non_mem_rec, subdir, 'IMR_mem_inj_non_mem_rec_result.json')
        res_non_mem_inj_mem_rec = _load_result(outdir_non_mem_inj_mem_rec, subdir, 'IMR_non_mem_inj_mem_rec_result.json')
        res_non_mem_inj_non_mem_rec = _load_result(outdir_non_mem_inj_non_mem_rec, subdir, 'IMR_non_mem_inj_non_mem_rec_result.json')

        try:
            sampling_bf = res_mem_inj_mem_rec.log_evidence - res_mem_inj_non_mem_rec.log_evidence
        except AttributeError as e:
            logger.warning(e)
            sampling_bf = np.nan
        logger.info('Parameter set: ' + str(subdir))
        logger.info('Sampling result log BF: \t' + str(sampling_bf))
        sampling_bfs.append(sampling_bf)

        settings.injection_parameters.__dict__ = memestr.core.submit.get_injection_parameter_set(id=99)
        waveform_generator_memory.parameters = settings.injection_parameters.__dict__
        waveform_generator_no_memory.parameters = settings.injection_parameters.__dict__

        logger.disabled = True
        ifos = [bb.gw.detector.get_interferometer_with_fake_noise_and_injection(
            name=name,
            injection_polarizations=waveform_generator_memory.frequency_domain_strain(),
            injection_parameters=settings.injection_parameters.__dict__,
            outdir=outdir,
            zero_noise=True,
            plot=False,
            **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors]
        logger.disabled = False

        likelihood.interferometers = bb.gw.detector.InterferometerList(ifos)
        likelihood.parameters = settings.injection_parameters.__dict__
        likelihood.waveform_generator = waveform_generator_memory
        evidence_memory = likelihood.log_likelihood()
        likelihood.waveform_generator = waveform_generator_no_memory
        evidence_non_memory = likelihood.log_likelihood()

        logger.info("Injected value log BF: \t" + str(evidence_memory - evidence_non_memory))
        injection_bfs.append(evidence_memory - evidence_non_memory)

        likelihood.waveform_generator = waveform_generator_memory
        try:
            log_weights = _calculate_log_weights(likelihood, res_mem_inj_non_mem_rec.posterior)
            reweighed_log_bf = reweigh_log_evidence_by_weights(res_mem_inj_non_mem_rec.log_evidence,
                                                               log_weights) - res_mem_inj_non_mem_rec.log_evidence
        except AttributeError as e:
            logger.warning(e)
            reweighed_log_bf = np.nan
        logger.info("Reweighed memory inj to memory log BF: \t" + str(reweighed_log_bf))
        reweighing_to_memory_bfs_mem_inj.append(reweighed_log_bf)

        likelihood.waveform_generator = waveform_generator_no_memory
        try:
            log_weights = _calculate_log_weights(likelihood, res_mem_inj_mem_rec.posterior)
            reweighed_log_bf = res_mem_inj_mem_rec.log_evidence - reweigh_log_evidence_by_weights(
                res_mem_inj_mem_rec.log_evidence,
                log_weights)
        except AttributeError as e:
            logger.warning(e)
            reweighed_log_bf = np.nan
        logger.info("Reweighed memory inj from memory log BF: \t" + str(reweighed_log_bf))
        reweighing_from_memory_bfs_mem_inj.append(reweighed_log_bf)

        likelihood.waveform_generator = waveform_generator_memory
        try:
            log_weights = _calculate_log_weights(likelihood, res_non_mem_inj_non_mem_rec.posterior)
            reweighed_log_bf = reweigh_log_evidence_by_weights(res_non_mem_inj_non_mem_rec.log_evidence,
                                                               log_weights) - res_non_mem_inj_non_mem_rec.log_evidence
        except AttributeError as e:
            logger.warning(e)
            reweighed_log_bf = np.nan
        logger.info("Reweighed non memory inj to memory log BF: \t" + str(reweighed_log_bf))
        reweighing_to_memory_bfs_non_mem_inj.append(reweighed_log_bf)

        likelihood.waveform_generator = waveform_generator_no_memory
        try:
            log_weights = _calculate_log_weights(likelihood, res_non_mem_inj_mem_rec.posterior)
            reweighed_log_bf = res_non_mem_inj_mem_rec.log_evidence - reweigh_log_evidence_by_weights(
                res_non_mem_inj_mem_rec.log_evidence,
                log_weights)
        except AttributeError as e:
            logger.warning(e)
            reweighed_log_bf = np.nan
        logger.info("Reweighed non memory inj from memory log BF: \t" + str(reweighed_log_bf))
        reweighing_from_memory_bfs_non_mem_inj.append(reweighed_log_bf)

    logger.info(np.sum(injection_bfs))
    logger.info(np.sum(sampling_bfs))
    logger.info(np.sum(reweighing_to_memory_bfs_mem_inj))
    logger.info(np.sum(reweighing_from_memory_bfs_mem_inj))
    logger.info(np.sum(reweighing_to_memory_bfs_non_mem_inj))
    logger.info(np.sum(reweighing_from_memory_bfs_non_mem_inj))
    res = pd.DataFrame.from_dict(dict(injection_bfs=injection_bfs,
                                      sampling_bfs=sampling_bfs,
                                      reweighing_to_memory_bfs_mem_inj=reweighing_to_memory_bfs_mem_inj,
                                      reweighing_from_memory_bfs_mem_inj=reweighing_from_memory_bfs_mem_inj,
                                      reweighing_to_memory_bfs_non_mem_inj=reweighing_to_memory_bfs_non_mem_inj,
                                      reweighing_from_memory_bfs_non_mem_inj=reweighing_from_memory_bfs_non_mem_inj))
    res.to_json('evidence_reweighing/' + str(subdirs[0]) + '_' + str(subdirs[-1]) + '.json')


def _load_result(outdir, subdir, label):
    try:
        res = bb.result.read_in_result(outdir + '/' + subdir + '/' + label)
    except OSError as e:
        logger.warning(e)
        res = None
    return res


def reweigh_log_evidence_by_weights(log_evidence, log_weights):
    return log_evidence + logsumexp(log_weights) - np.log(len(log_weights))


def _calculate_log_weights(likelihood, posterior):
    weights = []
    for i in range(len(posterior)):
        for parameter in ['total_mass', 'mass_ratio', 'inc', 'phase',
                          'ra', 'dec', 'psi', 'geocent_time']:
            likelihood.parameters[parameter] = posterior.iloc[i][parameter]
        weights.append(likelihood.log_likelihood_ratio() - posterior.iloc[i]['log_likelihood'])
    return weights


# Use sampling_frequency == 4096 from 0 to 64 and 2048 after that
# print_evidences(subdirs=[str(subdir) for subdir in range(int(sys.argv[1]), int(sys.argv[2]))],
#                 sampling_frequency=int(sys.argv[3]))
print_evidences(subdirs=[str(subdir) for subdir in range(10)], sampling_frequency=2048)
