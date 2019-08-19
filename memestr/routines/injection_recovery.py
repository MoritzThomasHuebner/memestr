from __future__ import division

from memestr.core.waveforms import *
from memestr.routines.setup import setup_run


def update_kwargs(default_kwargs, kwargs):
    new_kwargs = default_kwargs.copy()
    for key in list(set(default_kwargs.keys()).intersection(kwargs.keys())):
        new_kwargs[key] = kwargs[key]
    return new_kwargs


def run_production_injection(outdir, **kwargs):
    recovery_model = models[kwargs['recovery_model']]
    filename_base, ifos, likelihood, logger, priors, settings, sub_run_id = setup_run(
        kwargs, outdir, recovery_model)

    try:
        result = bilby.result.read_in_result(
            filename=str(filename_base) + '_production_IMR_non_mem_rec/' + str(sub_run_id) + 'IMR_mem_inj_non_mem_rec_result.json')
        result.outdir = outdir
    except Exception as e:
        logger.info(e)
        result = bilby.core.sampler.run_sampler(likelihood=likelihood,
                                                priors=priors,
                                                injection_parameters=deepcopy(settings.injection_parameters.__dict__),
                                                outdir=outdir,
                                                save=True,
                                                verbose=True,
                                                random_seed=np.random.randint(0, 100000),
                                                sampler=settings.sampler_settings.sampler,
                                                npoints=settings.sampler_settings.npoints,
                                                label=settings.sampler_settings.label,
                                                clean=settings.sampler_settings.clean,
                                                nthreads=settings.sampler_settings.nthreads,
                                                maxmcmc=settings.sampler_settings.maxmcmc,
                                                resume=settings.sampler_settings.resume,
                                                save_bounds=False,
                                                check_point_plot=False,
                                                walks=50)
                                                # n_check_point=20)
        result.save_to_file()
        logger.info(str(result))

    # try:
    #     result = bilby.result.read_in_result(
    #         filename=str(filename_base) + '_dynesty_production_IMR_non_mem_rec/' + sub_run_id + 'reconstructed_combined_result.json')
    # except Exception as e:
    #     logger.info(e)
    #     result.posterior = bilby.gw.conversion. \
    #         generate_posterior_samples_from_marginalized_likelihood(result.posterior, likelihood_imr_phenom)
    #     result.label = 'reconstructed_result' + sub_run_id
    #     result.save_to_file()

    params = deepcopy(settings.injection_parameters.__dict__)
    del params['s11']
    del params['s12']
    del params['s21']
    del params['s22']
    del params['random_injection_parameters']
    # result.plot_corner(lionize=settings.other_settings.lionize, parameters=params, outdir=outdir)


