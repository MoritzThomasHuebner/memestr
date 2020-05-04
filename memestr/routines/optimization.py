import bilby

from memestr.core.postprocessing import PostprocessingResult, adjust_phase_and_geocent_time_complete_posterior_parallel
from memestr.routines.setup import setup_run


def run_time_phase_optimization(**kwargs):
    filename_base, ifos, likelihood_imr_phenom, logger, priors, settings, sub_run_id = setup_run(kwargs)

    try:
        pp_result = PostprocessingResult.from_json(outdir=settings.sampler_settings.outdir)
    except Exception as e:
        logger.info(e)
        pp_result = PostprocessingResult(outdir=settings.sampler_settings.outdir)

    result = bilby.result.read_in_result(settings.sampler_settings.outdir + '/reconstructed_combined_result.json')
    if settings.sampler_settings.clean:
        time_and_phase_shifted_result = adjust_phase_and_geocent_time_complete_posterior_parallel(result, 16)
        time_and_phase_shifted_result.label = sub_run_id + 'time_and_phase_shifted_combined'
        time_and_phase_shifted_result.save_to_file()
    else:
        time_and_phase_shifted_result = bilby.result.read_in_result(
            filename=settings.sampler_settings.outdir + '/time_and_phase_shifted_combined_result.json')

    pp_result.to_json()
    return time_and_phase_shifted_result
