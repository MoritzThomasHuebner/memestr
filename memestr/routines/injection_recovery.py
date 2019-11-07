from __future__ import division

from memestr.core.waveforms import *
from memestr.routines.setup import setup_run


def update_kwargs(default_kwargs, kwargs):
    new_kwargs = default_kwargs.copy()
    for key in list(set(default_kwargs.keys()).intersection(kwargs.keys())):
        new_kwargs[key] = kwargs[key]
    return new_kwargs


def run_production_injection(**kwargs):
    filename_base, ifos, likelihood, logger, priors, settings, sub_run_id = setup_run(
        kwargs)

    result = bilby.core.sampler.run_sampler(likelihood=likelihood,
                                            priors=priors,
                                            injection_parameters=deepcopy(settings.injection_parameters.__dict__),
                                            save=True,
                                            verbose=True,
                                            save_bounds=False,
                                            check_point_plot=False,
                                            **settings.sampler_settings.__dict__)

    result.save_to_file()
    logger.info(str(result))

    if settings.sampler_settings.plot:
        params = deepcopy(settings.injection_parameters.__dict__)
        del params['s11']
        del params['s12']
        del params['s21']
        del params['s22']
        del params['random_injection_parameters']
        result.plot_corner(lionize=settings.other_settings.lionize, parameters=params)


