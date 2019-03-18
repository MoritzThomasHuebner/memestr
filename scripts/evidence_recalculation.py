from scipy.misc import logsumexp
import numpy as np
import bilby as bb
from memestr.core.parameters import AllSettings
from memestr.core.waveforms import time_domain_IMRPhenomD_waveform_with_memory, \
    time_domain_IMRPhenomD_waveform_without_memory
import sys
import pandas as pd

# parameter_set = 0
# parameter_set_dir = 'parameter_set_' + str(parameter_set)
distances = dict(a000=200, a001=230, a002=262, a003=299, a004=342, a005=391, a006=448, a007=512, a008=586, a009=670,
                 a010=766, a011=876, a012=1002, a013=1147, a014=1311, a015=1500)


run_id = sys.argv[1]
outdir = run_id + '_reweighing_result'
bb.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

parameters = dict(mass_ratio=0.8, total_mass=60.0, s11=0.0, s12=0.0, s13=0.0, s21=0.0, s22=0.0, s23=0.0,
                  luminosity_distance=distances['a' + run_id], inc=np.pi / 2, phase=1.3, ra=1.54,
                  dec=-0.7, psi=2.659, geocent_time=1126259642.413)
# outdir_mem_inj_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_mem_inj_mem_rec'
# outdir_mem_inj_non_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_mem_inj_non_mem_rec'
# outdir_non_mem_inj_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_non_mem_inj_mem_rec'
# outdir_non_mem_inj_non_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_non_mem_inj_non_mem_rec'
outdir_mem_inj_mem_rec = run_id + '_IMR_mem_inj_mem_rec/'
outdir_mem_inj_non_mem_rec = run_id + '_IMR_mem_inj_non_mem_rec/'

logger = bb.utils.logger

injection_bfs = []
sampling_bfs_mem_inj = []
sampling_bfs_non_mem_inj = []
reweighing_to_memory_bfs_mem_inj = []
reweighing_to_memory_bfs_non_mem_inj = []
reweighing_from_memory_bfs_mem_inj = []
reweighing_from_memory_bfs_non_mem_inj = []


def reweigh_evidences(subdirs, sampling_frequency=2048, duration=16, alpha=0.1):
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
                                    distance_marginalization=True,
                                    time_marginalization=True)

    for subdir in subdirs:
        res_mem_inj_mem_rec = _load_result(outdir_mem_inj_mem_rec, subdir, 'IMR_mem_inj_mem_rec_result.json')
        res_mem_inj_non_mem_rec = _load_result(outdir_mem_inj_non_mem_rec, subdir,
                                               'IMR_mem_inj_non_mem_rec_result.json')
        # res_non_mem_inj_mem_rec = _load_result(outdir_non_mem_inj_mem_rec, subdir, 'IMR_mem_inj_mem_rec_result.json')
        # res_non_mem_inj_non_mem_rec = _load_result(outdir_non_mem_inj_non_mem_rec, subdir, 'IMR_mem_inj_non_mem_rec_result.json')

        sampling_bf_mem_inj = _get_sampling_bf(res_mem_inj_mem_rec, res_mem_inj_non_mem_rec)
        # sampling_bf_non_mem_inj = _get_sampling_bf(res_non_mem_inj_mem_rec, res_non_mem_inj_non_mem_rec)
        logger.info('Run number: ' + str(subdir))
        logger.info('Sampling result memory injected log BF: \t' + str(sampling_bf_mem_inj))
        # logger.info('Sampling result no memory injected log BF: \t' + str(sampling_bf_non_mem_inj))

        sampling_bfs_mem_inj.append(sampling_bf_mem_inj)
        # sampling_bfs_non_mem_inj.append(sampling_bf_non_mem_inj)

        # settings.injection_parameters.__dict__ = memestr.core.submit.get_injection_parameter_set(id=parameter_set)
        settings.injection_parameters.__dict__ = parameters
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

        reweighed_debug_mem = _reweigh(likelihood, res_mem_inj_mem_rec, waveform_generator_memory)
        reweighed_debug_non_mem = -_reweigh(likelihood, res_mem_inj_non_mem_rec, waveform_generator_no_memory)
        logger.debug("Reweighed memory debug: \t" + str(reweighed_debug_mem))
        logger.debug("Reweighed no memory debug: \t" + str(reweighed_debug_non_mem))

        reweighed_log_bf_mem_inj_to_mem = _reweigh(likelihood, res_mem_inj_non_mem_rec, waveform_generator_memory)
        reweighed_log_bf_mem_inj_from_mem = -_reweigh(likelihood, res_mem_inj_mem_rec, waveform_generator_no_memory)
        # reweighed_log_bf_non_mem_inj_to_mem = _reweigh(likelihood, res_non_mem_inj_non_mem_rec, waveform_generator_memory)
        # reweighed_log_bf_non_mem_inj_from_mem = -_reweigh(likelihood, res_non_mem_inj_mem_rec, waveform_generator_no_memory)
        logger.info("Reweighed memory inj to memory log BF: \t" + str(reweighed_log_bf_mem_inj_to_mem))
        logger.info("Reweighed memory inj from memory log BF: \t" + str(reweighed_log_bf_mem_inj_from_mem))
        # logger.info("Reweighed non memory inj to memory log BF: \t" + str(reweighed_log_bf_non_mem_inj_to_mem))
        # logger.info("Reweighed non memory inj from memory log BF: \t" + str(reweighed_log_bf_non_mem_inj_from_mem))
        reweighing_to_memory_bfs_mem_inj.append(reweighed_log_bf_mem_inj_to_mem)
        reweighing_from_memory_bfs_mem_inj.append(reweighed_log_bf_mem_inj_from_mem)
        # reweighing_to_memory_bfs_non_mem_inj.append(reweighed_log_bf_non_mem_inj_to_mem)
        # reweighing_from_memory_bfs_non_mem_inj.append(reweighed_log_bf_non_mem_inj_from_mem)

    logger.info(np.sum(injection_bfs))
    logger.info(np.sum(sampling_bfs_mem_inj))
    logger.info(np.sum(reweighing_to_memory_bfs_mem_inj))
    logger.info(np.sum(reweighing_from_memory_bfs_mem_inj))
    logger.info(np.sum(reweighing_to_memory_bfs_non_mem_inj))
    logger.info(np.sum(reweighing_from_memory_bfs_non_mem_inj))
    res = pd.DataFrame.from_dict(dict(injection_bfs=injection_bfs,
                                      sampling_bfs=sampling_bfs_mem_inj,
                                      reweighing_to_memory_bfs_mem_inj=reweighing_to_memory_bfs_mem_inj,
                                      reweighing_from_memory_bfs_mem_inj=reweighing_from_memory_bfs_mem_inj,
                                      reweighing_to_memory_bfs_non_mem_inj=reweighing_to_memory_bfs_non_mem_inj,
                                      reweighing_from_memory_bfs_non_mem_inj=reweighing_from_memory_bfs_non_mem_inj))
    res.to_json('evidence_reweighing/' + str(subdirs[0]) + '_' + str(subdirs[-1]) + '.json')


def _get_sampling_bf(res_mem_inj_mem_rec, res_mem_inj_non_mem_rec):
    try:
        sampling_bf_mem_inj = res_mem_inj_mem_rec.log_evidence - res_mem_inj_non_mem_rec.log_evidence
    except AttributeError as e:
        logger.warning(e)
        sampling_bf_mem_inj = np.nan
    return sampling_bf_mem_inj


def _load_result(outdir, subdir, label):
    try:
        res = bb.result.read_in_result(outdir + '/' + subdir + '/' + label, extension='json')
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
        weights.append(likelihood.log_likelihood_ratio() - posterior.iloc[i]['log_likelihood'])
    return weights


# Use sampling_frequency == 4096 from 0 to 64 and 2048 after that for existing pop runs
# print_evidences(subdirs=[str(subdir) for subdir in range(int(sys.argv[1]), int(sys.argv[2]))],
#                 sampling_frequency=int(sys.argv[3]))
reweigh_evidences(subdirs=[str(subdir) for subdir in range(8)], sampling_frequency=2048)
