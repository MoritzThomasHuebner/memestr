from __future__ import division

import numpy as np
import bilby

from memestr.core.parameters import AllSettings, InjectionParameters


def run_basic_injection(injection_model, recovery_model, outdir, **kwargs):
    settings = AllSettings.from_defaults_with_some_specified_kwargs(**kwargs)
    np.random.seed(settings.other_settings.random_seed)
    print("Random seed: " + str(settings.other_settings.random_seed))

    bilby.core.utils.setup_logger(outdir=outdir, label=settings.sampler_settings.label)

    waveform_generator = bilby.gw.WaveformGenerator(time_domain_source_model=injection_model,
                                                    parameters=settings.injection_parameters.__dict__,
                                                    waveform_arguments=settings.waveform_arguments.__dict__,
                                                    **settings.waveform_data.__dict__)

    hf_signal = waveform_generator.frequency_domain_strain()
    ifos = [bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
        name,
        injection_polarizations=hf_signal,
        injection_parameters=settings.injection_parameters.__dict__,
        outdir=outdir,
        zero_noise=settings.detector_settings.zero_noise,
        **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors]

    waveform_generator.time_domain_source_model = recovery_model

    priors = settings.recovery_priors.proper_dict()
    likelihood = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=ifos,
                                    waveform_generator=waveform_generator,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization)

    result = bilby.core.sampler.run_sampler(likelihood=likelihood,
                                            priors=priors,
                                            injection_parameters=settings.injection_parameters.__dict__,
                                            outdir=outdir,
                                            save=False,
                                            verbose=True,
                                            **settings.sampler_settings.__dict__)
    result.plot_corner(lionize=settings.other_settings.lionize)
    result.memory_settings = repr(settings)

    print(result)
    result.save_to_file()
    return result


def update_kwargs(default_kwargs, kwargs):
    new_kwargs = default_kwargs.copy()
    for key in list(set(default_kwargs.keys()).intersection(kwargs.keys())):
        new_kwargs[key] = kwargs[key]
    return new_kwargs


def run_basic_injection_nrsur(injection_model, recovery_model, outdir, **kwargs):
    start_time = -0.5
    end_time = 0.0  # 0.029
    duration = end_time - start_time

    priors = dict()
    injection_parameters = InjectionParameters.init_with_updated_kwargs(**kwargs)
    for key in injection_parameters.__dict__:
        priors['prior_' + key] = injection_parameters.__dict__[key]
    # priors['prior_total_mass'] = bilby.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
    # priors['prior_mass_ratio'] = bilby.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$")
    # priors['prior_luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
    #                                                                            maximum=5e3, latex_label="$L_D$")
    # priors['prior_inc'] = bilby.core.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$")
    priors['prior_phase'] = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=np.pi, latex_label="$\phi$")
    # priors['prior_ra'] = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, latex_label="$RA$")
    # priors['prior_dec'] = bilby.core.prior.Cosine(name='dec', latex_label="$DEC$")
    priors['prior_psi'] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, latex_label="$\psi$")
    # priors['prior_geocent_time'] = bilby.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time')
    nr_sur_kwargs = dict(
        start_time=start_time,
        duration=duration,
        label='NRSur',
        new_seed=True,
        zero_noise=False
    )
    nr_sur_kwargs.update(priors)
    nr_sur_kwargs.update(kwargs)
    return run_basic_injection(injection_model=injection_model, recovery_model=recovery_model, outdir=outdir,
                               **nr_sur_kwargs)


def run_basic_injection_imr_phenom(injection_model, recovery_model, outdir, **kwargs):
    priors = dict()
    injection_parameters = InjectionParameters.init_with_updated_kwargs(**kwargs)
    for key in injection_parameters.__dict__:
        priors['prior_' + key] = injection_parameters.__dict__[key]
    # priors['prior_total_mass'] = bilby.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
    # priors['prior_mass_ratio'] = bilby.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$")
    # priors['prior_luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
    #                                                                            maximum=600, latex_label="$L_D$")
    # priors['prior_inc'] = bilby.core.prior.Sine(latex_label="$\iota$")
    # priors['prior_phase'] = bilby.core.prior.Uniform(name='phase', minimum=injection_parameters.phase - np.pi / 4,
    #                                                  maximum=injection_parameters.phase + np.pi / 4,
    #                                                  latex_label="$\phi$")
    # priors['prior_ra'] = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, latex_label="$RA$")
    # priors['prior_dec'] = bilby.core.prior.Cosine(name='dec', latex_label="$DEC$")
    # priors['prior_psi'] = bilby.core.prior.Uniform(name='psi', minimum=injection_parameters.psi - np.pi / 4,
    #                                                maximum=injection_parameters.psi + np.pi / 4, latex_label="$\psi$")
    # priors['prior_geocent_time'] = bilby.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time')

    #Distance vs evidence
    # priors['prior_total_mass'] = bilby.core.prior.Uniform(minimum=55, maximum=65.0, latex_label="$M_{tot}$")
    # priors['prior_mass_ratio'] = bilby.core.prior.Uniform(minimum=1.0, maximum=2.0, latex_label="$q$")
    # priors['prior_luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
    #                                                                            maximum=1000, latex_label="$L_D$")
    # priors['prior_inc'] = bilby.core.prior.Uniform(minimum=np.pi/2-0.25, maximum=np.pi/2+0.25, latex_label="$\iota$")
    # priors['prior_ra'] = bilby.core.prior.Uniform(name='ra', minimum=injection_parameters.ra-0.24,
    #                                               maximum=injection_parameters.ra+0.24, latex_label="$RA$")
    # priors['prior_dec'] = bilby.core.prior.Uniform(name='dec', minimum=injection_parameters.dec-0.2,
    #                                                maximum=injection_parameters.dec+0.2, latex_label="$DEC$")
    # priors['prior_phase'] = bilby.core.prior.Uniform(name='phase', minimum=injection_parameters.phase - np.pi/4,
    #                                                  maximum=injection_parameters.phase + np.pi/4, latex_label="$\phi$")
    # priors['prior_psi'] = bilby.core.prior.Uniform(name='psi', minimum=injection_parameters.psi - np.pi/4,
    #                                                maximum=injection_parameters.psi + np.pi/4, latex_label="$\psi$")
    # priors['prior_geocent_time'] = bilby.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time')

    priors['prior_total_mass'] = bilby.core.prior.Uniform(minimum=np.maximum(injection_parameters.total_mass - 20, 16),
                                                          maximum=injection_parameters.total_mass + 20,
                                                          latex_label="$M_{tot}$")
    priors['prior_mass_ratio'] = bilby.core.prior.Uniform(minimum=np.maximum(injection_parameters.mass_ratio-0.5, 0.4),
                                                          maximum=1,
                                                          latex_label="$q$")
    priors['prior_luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(minimum=10,
                                                                               maximum=2000,
                                                                               latex_label="$L_D$")
    priors['prior_inc'] = bilby.core.prior.Uniform(minimum=np.maximum(injection_parameters.inc-0.5, 0),
                                                   maximum=np.minimum(injection_parameters.inc+0.5, np.pi),
                                                   latex_label="$\iota$")
    priors['prior_ra'] = bilby.core.prior.Uniform(minimum=np.maximum(injection_parameters.ra-0.6, 0),
                                                  maximum=np.minimum(injection_parameters.ra+0.6, 2*np.pi),
                                                  latex_label="$RA$")
    priors['prior_dec'] = bilby.core.prior.Uniform(minimum=np.maximum(injection_parameters.dec-0.5, -np.pi/2),
                                                   maximum=np.minimum(injection_parameters.dec+0.5, np.pi/2),
                                                   latex_label="$DEC$")
    priors['prior_phase'] = bilby.core.prior.Uniform(minimum=np.maximum(injection_parameters.phase - np.pi/4, 0),
                                                     maximum=np.minimum(injection_parameters.phase + np.pi/4, 2*np.pi),
                                                     latex_label="$\phi$")
    priors['prior_psi'] = bilby.core.prior.Uniform(minimum=np.maximum(injection_parameters.psi - np.pi/4, 0),
                                                   maximum=np.minimum(injection_parameters.psi + np.pi/4, 2*np.pi),
                                                   latex_label="$\psi$")
    priors['prior_geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters.geocent_time - 0.1,
                                                            maximum=injection_parameters.geocent_time + 0.1,
                                                            latex_label='$t_c$')

    imr_phenom_kwargs = dict(
        label='IMRPhenomD',
        npoints=5000
    )
    imr_phenom_kwargs.update(priors)
    imr_phenom_kwargs.update(kwargs)
    return run_basic_injection(injection_model=injection_model, recovery_model=recovery_model, outdir=outdir,
                               **imr_phenom_kwargs)
