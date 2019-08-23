from copy import deepcopy

import bilby

from memestr.core.postprocessing import PostprocessingResult, reweigh_by_likelihood
from memestr.core.waveforms import models
from memestr.routines.setup import setup_run


def run_reweighting(outdir, **kwargs):
    recovery_model = models[kwargs['recovery_model']]
    reweight_model = models[kwargs['reweight_model']]
    filename_base, ifos, likelihood, logger, priors, settings, sub_run_id = setup_run(
        kwargs, outdir, recovery_model)
    try:
        raise Exception
        pp_result = PostprocessingResult.from_json(outdir=str(filename_base) + '_production_IMR_non_mem_rec/',
                                                   filename=str(sub_run_id) + 'pp_result.json')
    except Exception as e:
        logger.info(e)
        pp_result = PostprocessingResult(outdir=str(filename_base) + '_production_IMR_non_mem_rec/'.format(sub_run_id),
                                         filename=str(sub_run_id) + 'pp_result.json')
    #IMR
    # result = bilby.result.read_in_result(filename=str(filename_base) + '_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(sub_run_id))
    # time_and_phase_shifted_result = bilby.result.read_in_result(filename=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/time_and_phase_shifted_combined_result.json')
    # try:
    result = bilby.result.read_in_result(filename=str(filename_base) + '_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(sub_run_id))
    # result = bilby.result.read_in_result(filename=str(filename_base) + '_production_IMR_non_mem_rec/combined_proper_prior_result.json')
    # except Exception:
    #     result = bilby.result.read_in_result(filename=str(filename_base) + '_production_IMR_non_mem_rec/combined_proper_prior_result.json')

    if reweight_model.__name__.startswith('frequency'):
        waveform_generator_reweight = bilby.gw.WaveformGenerator(
            frequency_domain_source_model=reweight_model,
            parameters=deepcopy(settings.injection_parameters.__dict__),
            waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
            **settings.waveform_data.__dict__)
    else:
        waveform_generator_reweight = bilby.gw.WaveformGenerator(
            time_domain_source_model=reweight_model,
            parameters=deepcopy(settings.injection_parameters.__dict__),
            waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
            **settings.waveform_data.__dict__)

    if recovery_model.__name__.startswith('frequency'):
        waveform_generator_recovery = bilby.gw.WaveformGenerator(
            frequency_domain_source_model=recovery_model,
            parameters=deepcopy(settings.injection_parameters.__dict__),
            waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
            **settings.waveform_data.__dict__)
    else:
        waveform_generator_recovery = bilby.gw.WaveformGenerator(
            time_domain_source_model=recovery_model,
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

    # if pp_result.memory_weights is None:
    # memory_hom_log_bf, memory_hom_weights = reweigh_by_likelihood_parallel(new_likelihood=likelihood_reweight,
    #                                                                        new_result=time_and_phase_shifted_result,
    #                                                                        reference_likelihood=likelihood_imr_phenom_unmarginalized,
    #                                                                        reference_result=result,
    #                                                                        n_parallel=16)
    # memory_log_bf = memory_hom_log_bf - pp_result.hom_log_bf

    reweight_log_bf, weights = reweigh_by_likelihood(new_likelihood=likelihood_reweight,
                                                     new_result=result,
                                                     reference_likelihood=likelihood_recovery)
    memory_log_bf = reweight_log_bf
    pp_result.memory_log_bf = memory_log_bf
    pp_result.memory_weights = weights
    pp_result.to_json()
    logger.info("MEMORY LOG BF: " + str(pp_result.memory_log_bf))