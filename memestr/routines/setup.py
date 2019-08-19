import logging
import time
from copy import deepcopy

import bilby
import numpy as np

from memestr.core.parameters import AllSettings
from memestr.core.submit import get_injection_parameter_set


def setup_run(kwargs, outdir, recovery_model):
    priors = dict()
    filename_base = str(kwargs.get('filename_base', 0))
    filename_base = filename_base.replace('_dynesty', '')
    filename_base = filename_base.replace('_cpnest', '')
    filename_base = filename_base.replace('_pypolychord', '')
    injection_params_file = filename_base.replace('_nr_sur', '')
    injection_params_file = injection_params_file.replace('_IMR_inj', '')
    injection_parameters = get_injection_parameter_set(injection_params_file)
    for key in injection_parameters:
        priors['prior_' + key] = injection_parameters[key]
    priors['prior_total_mass'] = bilby.core.prior.Uniform(
        minimum=np.maximum(injection_parameters['total_mass'] - 20, 10),
        maximum=injection_parameters['total_mass'] + 30,
        latex_label="$M_{tot}$")
    priors['prior_mass_ratio'] = bilby.core.prior.Uniform(
        minimum=0.125,
        maximum=1,
        latex_label="$q$")
    priors['prior_luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(minimum=10,
                                                                               maximum=5000,
                                                                               latex_label="$L_D$",
                                                                               name='luminosity_distance')
    priors['prior_inc'] = bilby.core.prior.Sine(latex_label="$\\theta_{jn}$")
    priors['prior_ra'] = bilby.core.prior.Uniform(minimum=0, maximum=2 * np.pi, latex_label="$RA$")
    priors['prior_dec'] = bilby.core.prior.Cosine(latex_label="$DEC$")
    # priors['prior_phase'] = bilby.core.prior.Uniform(minimum=0,
    #                                                  maximum=2*np.pi,
    #                                                  latex_label="$\phi$")
    # priors['prior_psi'] = bilby.core.prior.Uniform(minimum=0,
    #                                                maximum=np.pi,
    #                                                latex_label="$\psi$")
    priors['prior_phase'] = bilby.core.prior.Uniform(minimum=injection_parameters['phase'] - np.pi / 2,
                                                     maximum=injection_parameters['phase'] + np.pi / 2,
                                                     latex_label="$\phi$")
    priors['prior_psi'] = bilby.core.prior.Uniform(minimum=injection_parameters['psi'] - np.pi / 2,
                                                   maximum=injection_parameters['psi'] + np.pi / 2,
                                                   latex_label="$\psi$")
    priors['prior_geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters['geocent_time'] - 0.1,
                                                            maximum=injection_parameters['geocent_time'] + 0.1,
                                                            latex_label='$t_c$')
    priors['prior_s13'] = bilby.gw.prior.AlignedSpin(name='s13', a_prior=bilby.core.prior.Uniform(0.0, 0.5),
                                                     latex_label='s13')
    priors['prior_s23'] = bilby.gw.prior.AlignedSpin(name='s23', a_prior=bilby.core.prior.Uniform(0.0, 0.5),
                                                     latex_label='s23')
    imr_phenom_kwargs = dict(
        label='IMRPhenomD'
    )
    imr_phenom_kwargs.update(priors)
    imr_phenom_kwargs.update(kwargs)
    logger = logging.getLogger('bilby')
    settings = AllSettings.from_defaults_with_some_specified_kwargs(**imr_phenom_kwargs, **injection_parameters)
    settings.waveform_data.start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
    sub_run_id = str(kwargs.get('sub_run_id', ''))
    settings.sampler_settings.label = sub_run_id + settings.sampler_settings.label
    bilby.core.utils.setup_logger(outdir=outdir, label=settings.sampler_settings.label)
    logger.info("Sub run ID: " + str(sub_run_id))
    # if sub_run_id == '':
    #     sys.exit(1)
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