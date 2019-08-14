import bilby

from memestr.core.postprocessing import PostprocessingResult, adjust_phase_and_geocent_time_complete_posterior_parallel, \
    adjust_phase_and_geocent_time_complete_posterior_proper
from memestr.core.waveforms import models
from memestr.routines.setup import setup_run


def run_time_phase_optimization(outdir, **kwargs):
    recovery_model = models[kwargs['recovery_model']]
    filename_base, ifos, likelihood_imr_phenom, likelihood_imr_phenom_unmarginalized, logger, priors, settings, sub_run_id = setup_run(
        kwargs, outdir, recovery_model)
    try:
        pp_result = PostprocessingResult.from_json(str(filename_base) + '_dynesty_production_IMR_non_mem_rec/')
    except Exception as e:
        logger.info(e)
        pp_result = PostprocessingResult(outdir=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/')
    result = bilby.result.read_in_result(filename=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/reconstructed_combined_result.json')
    try:
        # raise Exception
        time_and_phase_shifted_result = bilby.result.read_in_result(
            filename=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/time_and_phase_shifted_combined_result.json')
        # maximum_overlaps = pp_result.maximum_overlaps
    except Exception as e:
        logger.warning(e)
        time_and_phase_shifted_result = adjust_phase_and_geocent_time_complete_posterior_parallel(result, 16)
        # pp_result.maximum_overlaps = maximum_overlaps
        time_and_phase_shifted_result.label = sub_run_id + 'time_and_phase_shifted_combined'
        time_and_phase_shifted_result.save_to_file()
        # time_and_phase_shifted_result.plot_corner(parameters=deepcopy(params), outdir=outdir)
    pp_result.to_json()
    return time_and_phase_shifted_result


def run_time_phase_optimization_debug(recovery_model, outdir, **kwargs):
    filename_base, ifos, likelihood_imr_phenom, likelihood_imr_phenom_unmarginalized, logger, priors, settings, sub_run_id = setup_run(
        kwargs, outdir, recovery_model)
    result = bilby.result.read_in_result(filename=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/reconstructed_combined_result.json')
    return adjust_phase_and_geocent_time_complete_posterior_proper(result, ifos[0], True)