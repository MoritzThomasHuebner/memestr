import logging
import time
from copy import deepcopy

import bilby
import numpy as np

from memestr.core.parameters import AllSettings
from memestr.core.submit import get_injection_parameter_set


def setup_run(kwargs):
    recovery_model = kwargs['recovery_model']
    logger = logging.getLogger('bilby')

    filename_base = kwargs['outdir']

    injection_parameters = get_injection_parameter_set(filename_base)
    priors = _setup_priors(injection_parameters, kwargs)

    settings = AllSettings.from_defaults_with_some_specified_kwargs(**kwargs, **injection_parameters)
    settings.waveform_data.start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
    sub_run_id = str(kwargs.get('sub_run_id', ''))
    settings.sampler_settings.label = sub_run_id + settings.sampler_settings.label

    logger.info("Sub run ID: " + str(sub_run_id))
    logger.info("Parameter Set: " + str(filename_base))
    ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(filename_base) + '_H1L1V1.h5')
    for ifo in ifos:
        setattr(ifo.strain_data, '_frequency_mask_updated', True)

    waveform_generator = bilby.gw.WaveformGenerator(frequency_domain_source_model=recovery_model,
                                                    parameters=injection_parameters,
                                                    waveform_arguments=settings.waveform_arguments.__dict__,
                                                    **settings.waveform_data.__dict__)

    likelihood = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator,
                                    priors=priors,
                                    time_marginalization=kwargs['time_marginalization'],
                                    distance_marginalization=kwargs['distance_marginalization'],
                                    phase_marginalization=kwargs['phase'])

    likelihood.parameters = deepcopy(settings.injection_parameters.__dict__)
    np.random.seed(int(time.time() * 1000000) % 1000000)
    logger.info('Injection Parameters:')
    logger.info(str(settings.injection_parameters))
    logger.info('Settings:')
    logger.info(str(settings))
    return filename_base, ifos, likelihood, logger, priors, settings, sub_run_id


def _setup_priors(injection_parameters, kwargs):
    priors = kwargs['priors']
    priors['phase'] = bilby.core.prior.Uniform(minimum=injection_parameters['phase'] - np.pi / 4,
                                               maximum=injection_parameters['phase'] + np.pi / 4,
                                               name="phase")
    priors['psi'] = bilby.core.prior.Uniform(minimum=injection_parameters['psi'] - np.pi / 4,
                                             maximum=injection_parameters['psi'] + np.pi / 4,
                                             latex_label="polarisation")
    priors['geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters['geocent_time'] - 0.1,
                                                      maximum=injection_parameters['geocent_time'] + 0.1,
                                                      latex_label='$t_c$')
    return priors
