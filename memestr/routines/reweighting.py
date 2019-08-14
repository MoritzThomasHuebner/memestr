from copy import deepcopy

import bilby

from memestr.core.postprocessing import PostprocessingResult, reweigh_by_likelihood
from memestr.core.waveforms import time_domain_IMRPhenomD_waveform_with_memory, \
    time_domain_IMRPhenomD_waveform_without_memory, models
from memestr.routines.setup import setup_run


def run_reweighting(outdir, **kwargs):
    recovery_model = models[kwargs['recovery_model']]
    filename_base, ifos, likelihood_imr_phenom, likelihood_imr_phenom_unmarginalized, logger, priors, settings, sub_run_id = setup_run(
        kwargs, outdir, recovery_model)
    try:
        pp_result = PostprocessingResult.from_json(outdir=str(filename_base) + '_production_IMR_non_mem_rec/',
                                                   filename=str(sub_run_id) + 'pp_result.json')
    except Exception as e:
        logger.info(e)
        pp_result = PostprocessingResult(outdir=str(filename_base) + '_production_IMR_non_mem_rec/'.format(sub_run_id),
                                         filename=str(sub_run_id) + 'pp_result.json')
    #IMR
    result = bilby.result.read_in_result(filename=str(filename_base) + '_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(sub_run_id))
    # time_and_phase_shifted_result = bilby.result.read_in_result(filename=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/time_and_phase_shifted_combined_result.json')
    # try:
    #     result = bilby.result.read_in_result(filename=str(filename_base) + '_production_IMR_non_mem_rec/combined_high_nlive_result.json')
    # except Exception:
    #     result = bilby.result.read_in_result(filename=str(filename_base) + '_production_IMR_non_mem_rec/combined_result.json')

    # waveform_generator_memory = bilby.gw.WaveformGenerator(
    #     frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
    #     parameters=deepcopy(settings.injection_parameters.__dict__),
    #     waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
    #     **settings.waveform_data.__dict__)
    #
    # waveform_generator_no_memory = bilby.gw.WaveformGenerator(
    #     frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped,
    #     parameters=deepcopy(settings.injection_parameters.__dict__),
    #     waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
    #     **settings.waveform_data.__dict__)
    waveform_generator_memory = bilby.gw.WaveformGenerator(
        time_domain_source_model=time_domain_IMRPhenomD_waveform_with_memory,
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)

    waveform_generator_no_memory = bilby.gw.WaveformGenerator(
        time_domain_source_model=time_domain_IMRPhenomD_waveform_without_memory,
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)

    likelihood_memory = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_memory,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization)

    likelihood_no_memory = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_no_memory,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization)
    #
    # result = bilby.result.read_in_result(filename=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/IMR_mem_inj_non_mem_rec_result.json')
    # waveform_generator_memory = bilby.gw.WaveformGenerator(
    #     frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped_lm_modes,
    #     parameters=deepcopy(settings.injection_parameters.__dict__),
    #     waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
    #     **settings.waveform_data.__dict__)
    #
    # waveform_generator_no_memory = bilby.gw.WaveformGenerator(
    #     frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_lm_modes,
    #     parameters=deepcopy(settings.injection_parameters.__dict__),
    #     waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
    #     **settings.waveform_data.__dict__)

    # likelihood_memory = HOMTimePhaseMarginalizedGWT(interferometers=deepcopy(ifos),
    #                                                 waveform_generator=waveform_generator_memory,
    #                                                 priors=priors)
    #
    # likelihood_no_memory = HOMTimePhaseMarginalizedGWT(interferometers=deepcopy(ifos),
    #                                                    waveform_generator=waveform_generator_no_memory,
    #                                                    priors=priors)
    #
    likelihood_no_memory.parameters = deepcopy(settings.injection_parameters.__dict__)
    likelihood_memory.parameters = deepcopy(settings.injection_parameters.__dict__)

    # if True:
        # hom_log_bf, hom_weights = reweigh_by_likelihood(new_likelihood=likelihood_no_memory,
        #                                                 new_result=result,
        #                                                 reference_likelihood=likelihood_imr_phenom_unmarginalized,
        #                                                 reference_result=result)
        # hom_log_bf, hom_weights = reweigh_by_likelihood_parallel(new_likelihood=likelihood_no_memory,
        #                                                          new_result=time_and_phase_shifted_result,
        #                                                          reference_likelihood=likelihood_imr_phenom_unmarginalized,
        #                                                          reference_result=result,
        #                                                          n_parallel=16
        #                                                          )
        # pp_result.hom_weights = hom_weights
        # pp_result.hom_log_bf = hom_log_bf
        # pp_result.to_json()

    # logger.info("HOM LOG BF:" + str(pp_result.hom_log_bf))
    # logger.info("Number of weights:" + str(len(pp_result.hom_weights)))
    # logger.info("Number of effective samples:" + str(pp_result.effective_samples))

    # if pp_result.memory_weights is None:
    if True:
        # IMR rec
        # memory_hom_log_bf, memory_hom_weights = reweigh_by_likelihood_parallel(new_likelihood=likelihood_memory,
        #                                                                        new_result=time_and_phase_shifted_result,
        #                                                                        reference_likelihood=likelihood_imr_phenom_unmarginalized,
        #                                                                        reference_result=result,
        #                                                                        n_parallel=16)
        # memory_log_bf = memory_hom_log_bf - pp_result.hom_log_bf

        # NRSur rec
        memory_hom_log_bf, memory_hom_weights = reweigh_by_likelihood(new_likelihood=likelihood_memory,
                                                                      new_result=result,
                                                                      reference_likelihood=likelihood_no_memory)
        memory_log_bf = memory_hom_log_bf
        pp_result.memory_log_bf = memory_log_bf
        pp_result.memory_weights = memory_hom_weights
        pp_result.to_json()
    logger.info("MEMORY LOG BF: " + str(pp_result.memory_log_bf))