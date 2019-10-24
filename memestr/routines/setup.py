import logging
import time
from copy import deepcopy

import bilby
import numpy as np

from memestr.core.parameters import AllSettings
from memestr.core.submit import get_injection_parameter_set


def setup_run(kwargs, outdir, recovery_model):
    logger = logging.getLogger('bilby')

    filename_base = str(kwargs.get('filename_base', 0))
    filename_base = filename_base.replace('_dynesty', '')
    filename_base = filename_base.replace('_cpnest', '')
    filename_base = filename_base.replace('_pypolychord', '')
    injection_params_file = filename_base.replace('_nr_sur', '')
    injection_params_file = injection_params_file.replace('_IMR_inj', '')
    injection_parameters = get_injection_parameter_set(injection_params_file)
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
    imr_phenom_kwargs = dict(
        label='IMRPhenomD'
    )
    imr_phenom_kwargs.update(priors)
    imr_phenom_kwargs.update(kwargs)
    settings = AllSettings.from_defaults_with_some_specified_kwargs(**imr_phenom_kwargs, **injection_parameters)
    settings.waveform_data.start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
    sub_run_id = str(kwargs.get('sub_run_id', ''))
    settings.sampler_settings.label = sub_run_id + settings.sampler_settings.label
    bilby.core.utils.setup_logger(outdir=outdir, label=settings.sampler_settings.label)
    logger.info("Sub run ID: " + str(sub_run_id))

    filename_base = str(kwargs.get('filename_base', 0))
    ifo_file = filename_base.replace('_dynesty', '')
    ifo_file = ifo_file.replace('_cpnest', '')
    ifo_file = ifo_file.replace('_pypolychord', '')
    ifo_file = ifo_file.replace('_nr_sur', '')
    ifo_file = ifo_file.replace('_IMR_inj', '')

    logger.info("Parameter Set: " + str(filename_base))
    ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' +
                                                          str(ifo_file) +
                                                          '_H1L1V1.h5')
    for ifo in ifos:
        setattr(ifo.strain_data, '_frequency_mask_updated', True)

    if recovery_model.__name__.startswith('frequency'):
        waveform_generator = bilby.gw.WaveformGenerator(frequency_domain_source_model=recovery_model,
                                                        parameters=settings.injection_parameters.__dict__,
                                                        waveform_arguments=settings.waveform_arguments.__dict__,
                                                        **settings.waveform_data.__dict__)
    else:
        waveform_generator = bilby.gw.WaveformGenerator(time_domain_source_model=recovery_model,
                                                        parameters=settings.injection_parameters.__dict__,
                                                        waveform_arguments=settings.waveform_arguments.__dict__,
                                                        **settings.waveform_data.__dict__)

    priors = deepcopy(settings.recovery_priors.proper_dict())
    likelihood = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization,
                                    distance_marginalization_lookup_table=[])
    likelihood.parameters = deepcopy(settings.injection_parameters.__dict__)
    np.random.seed(int(time.time() * 1000000) % 1000000)
    logger.info('Injection Parameters')
    logger.info(str(settings.injection_parameters))
    return filename_base, ifos, likelihood, logger, priors, settings, sub_run_id