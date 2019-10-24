from copy import deepcopy

import bilby

from memestr.core.postprocessing import PostprocessingResult, reweigh_by_likelihood
from memestr.routines.setup import setup_run


def run_reweighting(**kwargs):
    filename_base, ifos, likelihood, logger, priors, settings, sub_run_id = setup_run(
        kwargs)
    outdir = settings.sampler_settings.outdir
    try:
        pp_result = PostprocessingResult.from_json(outdir=outdir,
                                                   filename=str(sub_run_id) + 'pp_result.json')
    except Exception as e:
        logger.info(e)
        pp_result = PostprocessingResult(outdir=outdir,
                                         filename=str(sub_run_id) + 'pp_result.json')
    result = bilby.result.read_in_result(filename=outdir + '/{}IMR_mem_inj_non_mem_rec_result.json'.format(sub_run_id))

    waveform_generator_reweight = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=kwargs['reweight_model'],
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)

    waveform_generator_recovery = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=kwargs['recovery_model'],
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)

    likelihood_reweight = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_reweight,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization,
                                    distance_marginalization_lookup_table=[])

    likelihood_recovery = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_recovery,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization,
                                    distance_marginalization_lookup_table=[])

    likelihood_recovery.parameters = deepcopy(settings.injection_parameters.__dict__)
    likelihood_reweight.parameters = deepcopy(settings.injection_parameters.__dict__)

    reweight_log_bf, weights = reweigh_by_likelihood(new_likelihood=likelihood_reweight,
                                                     new_result=result,
                                                     reference_likelihood=likelihood_recovery)
    memory_log_bf = reweight_log_bf
    pp_result.memory_log_bf = memory_log_bf
    pp_result.memory_weights = weights
    pp_result.to_json()
    logger.info("MEMORY LOG BF: " + str(pp_result.memory_log_bf))
