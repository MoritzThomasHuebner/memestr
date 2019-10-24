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
        kwargs, recovery_model)

    result = bilby.core.sampler.run_sampler(likelihood=likelihood,
                                            priors=priors,
                                            injection_parameters=deepcopy(settings.injection_parameters.__dict__),
                                            outdir=outdir,
                                            save=True,
                                            verbose=True,
                                            sampler=settings.sampler_settings.sampler,
                                            npoints=settings.sampler_settings.npoints,
                                            label=settings.sampler_settings.label,
                                            clean=settings.sampler_settings.clean,
                                            resume=settings.sampler_settings.resume,
                                            save_bounds=False,
                                            check_point_plot=False,
                                            walks=settings.sampler_settings.walks)
    result.save_to_file()
    logger.info(str(result))

    if settings.sampler_settings.plot:
        params = deepcopy(settings.injection_parameters.__dict__)
        del params['s11']
        del params['s12']
        del params['s21']
        del params['s22']
        del params['random_injection_parameters']
        result.plot_corner(lionize=settings.other_settings.lionize, parameters=params, outdir=outdir)


