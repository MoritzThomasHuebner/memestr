import numpy as np
import bilby as bb
from memestr.core.parameters import AllSettings
from memestr.core.postprocessing import reweigh_by_likelihood
from memestr.core.waveforms import time_domain_IMRPhenomD_waveform_with_memory, \
    time_domain_IMRPhenomD_waveform_without_memory
import sys
from copy import deepcopy
import pandas as pd
from memestr.core.utils import get_ifo

# parameter_set = 0
# parameter_set_dir = 'parameter_set_' + str(parameter_set)
distances = dict(a032=200, a033=230, a034=262, a035=299, a036=342, a037=391, a038=448, a039=512, a040=586, a041=670,
                 a042=766, a043=876, a044=1002, a045=1147, a046=1311, a047=1500)

run_id = sys.argv[1]
outdir = run_id + '_reweighing_result'
bb.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

parameters = dict(mass_ratio=0.8, total_mass=60.0, s11=0.0, s12=0.0, s13=0.0, s21=0.0, s22=0.0, s23=0.0,
                  luminosity_distance=distances['a' + run_id], inc=np.pi / 2, phase=1.3, ra=1.54,
                  dec=-0.7, psi=0.9, geocent_time=1126259642.413)
# outdir_mem_inj_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_mem_inj_mem_rec'
# outdir_mem_inj_non_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_mem_inj_non_mem_rec'
# outdir_non_mem_inj_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_non_mem_inj_mem_rec'
# outdir_non_mem_inj_non_mem_rec = 'evidence_reweighing/' + parameter_set_dir + '/IMR_non_mem_inj_non_mem_rec'
outdir_mem_inj_mem_rec = run_id + '_IMR_mem_inj_mem_rec'
outdir_mem_inj_non_mem_rec = run_id + '_IMR_mem_inj_non_mem_rec'

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
    settings.injection_parameters.__dict__ = deepcopy(parameters)
    settings.detector_settings.zero_noise = True
    settings.waveform_data.start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
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
    hf_signal = waveform_generator_memory.frequency_domain_strain()
    ifos = [get_ifo(hf_signal, name, outdir, settings, waveform_generator_memory)
            for name in settings.detector_settings.detectors]
    ifos = bb.gw.detector.InterferometerList(ifos)
    logger.disabled = False
    priors = dict(luminosity_distance=bb.gw.prior.UniformComovingVolume(name='luminosity_distance',
                                                                        minimum=10,
                                                                        maximum=5000),
                  geocent_time=bb.core.prior.Uniform(minimum=settings.injection_parameters.geocent_time - 0.1,
                                                     maximum=settings.injection_parameters.geocent_time + 0.1,
                                                     name='geocent_time'))
    likelihood = bb.gw.likelihood \
        .GravitationalWaveTransient(interferometers=ifos,
                                    waveform_generator=waveform_generator_memory,
                                    priors=priors,
                                    distance_marginalization=True,
                                    time_marginalization=False)

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
        settings.injection_parameters.__dict__ = deepcopy(parameters)
        waveform_generator_memory.parameters = deepcopy(parameters)
        waveform_generator_no_memory.parameters = deepcopy(parameters)

        logger.disabled = True
        # ifos = [bb.gw.detector.get_interferometer_with_fake_noise_and_injection(
        #     name=name,
        #     injection_polarizations=waveform_generator_memory.frequency_domain_strain(),
        #     injection_parameters=settings.injection_parameters.__dict__,
        #     outdir=outdir,
        #     zero_noise=True,
        #     plot=False,
        #     **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors]
        ifos = [get_ifo(hf_signal, name, outdir, settings, waveform_generator_memory)
                for name in settings.detector_settings.detectors]
        logger.disabled = False
        likelihood.interferometers = bb.gw.detector.InterferometerList(ifos)
        likelihood.parameters = deepcopy(parameters)

        likelihood.waveform_generator = waveform_generator_no_memory
        evidence_non_memory = likelihood.log_likelihood_ratio()

        likelihood.waveform_generator = waveform_generator_memory
        evidence_memory = likelihood.log_likelihood_ratio()
        logger.info("Injected value log BF: \t" + str(evidence_memory - evidence_non_memory))
        injection_bfs.append(evidence_memory - evidence_non_memory)

        likelihood.waveform_generator = waveform_generator_memory
        reweighed_log_bf_mem_to_self, _ = reweigh_by_likelihood(likelihood, res_mem_inj_mem_rec, waveform_generator_memory)
        logger.info("Reweighed memory to self: \t" + str(reweighed_log_bf_mem_to_self))

        likelihood.waveform_generator = waveform_generator_no_memory
        reweighed_log_bf_non_mem_to_self, _ = -reweigh_by_likelihood(likelihood, res_mem_inj_non_mem_rec)
        logger.info("Reweighed no memory to self: \t" + str(reweighed_log_bf_non_mem_to_self))

        likelihood.waveform_generator = waveform_generator_memory
        reweighed_log_bf_mem_inj_to_mem, _ = reweigh_by_likelihood(likelihood, res_mem_inj_non_mem_rec)
        logger.info("Reweighed memory inj to memory log BF: \t" + str(reweighed_log_bf_mem_inj_to_mem +
                                                                      reweighed_log_bf_non_mem_to_self))
        likelihood.waveform_generator = waveform_generator_memory
        reweighed_log_bf_mem_inj_from_mem, _ = -reweigh_by_likelihood(likelihood, res_mem_inj_mem_rec)
        logger.info("Reweighed memory inj from memory log BF: \t" + str(reweighed_log_bf_mem_inj_from_mem +
                                                                        reweighed_log_bf_mem_to_self))

        reweighing_to_memory_bfs_mem_inj.append(reweighed_log_bf_mem_inj_to_mem + reweighed_log_bf_non_mem_to_self)
        reweighing_from_memory_bfs_mem_inj.append(reweighed_log_bf_mem_inj_from_mem + reweighed_log_bf_mem_to_self)


    logger.info(np.sum(injection_bfs))
    logger.info(np.sum(sampling_bfs_mem_inj))
    logger.info(np.sum(reweighing_to_memory_bfs_mem_inj))
    logger.info(np.sum(reweighing_from_memory_bfs_mem_inj))
    logger.info(str(injection_bfs))
    logger.info(str(sampling_bfs_mem_inj))
    logger.info(str(reweighing_to_memory_bfs_mem_inj))
    logger.info(str(reweighing_from_memory_bfs_mem_inj))
    # logger.info(np.sum(reweighing_to_memory_bfs_non_mem_inj))
    # logger.info(np.sum(reweighing_from_memory_bfs_non_mem_inj))
    res = pd.DataFrame.from_dict(dict(injection_bfs=injection_bfs,
                                      sampling_bfs=sampling_bfs_mem_inj,
                                      reweighing_to_memory_bfs_mem_inj=reweighing_to_memory_bfs_mem_inj,
                                      reweighing_from_memory_bfs_mem_inj=reweighing_from_memory_bfs_mem_inj))
                                      # reweighing_to_memory_bfs_non_mem_inj=reweighing_to_memory_bfs_non_mem_inj,
                                      # reweighing_from_memory_bfs_non_mem_inj=reweighing_from_memory_bfs_non_mem_inj))
    res.to_json(outdir + '/' + str(subdirs[0]) + '_' + str(subdirs[-1]) + '.json')


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


# Use sampling_frequency == 4096 from 0 to 64 and 2048 after that for existing pop runs
# print_evidences(subdirs=[str(subdir) for subdir in range(int(sys.argv[1]), int(sys.argv[2]))],
#                 sampling_frequency=int(sys.argv[3]))
reweigh_evidences(subdirs=[str(subdir) for subdir in range(0, 8)])