from __future__ import division
import numpy as np
import tupak

from memestr.submit.parameters import AllSettings, InjectionParameters


def run_basic_injection(injection_model, recovery_model, outdir, **kwargs):
    settings = AllSettings.from_defaults_with_some_specified_kwargs(**kwargs)

    tupak.core.utils.setup_logger(outdir=outdir, label=settings.sampler_settings.label)
    if not settings.other_settings.new_seed:
        np.random.seed(88170235)

    waveform_generator = tupak.gw.WaveformGenerator(time_domain_source_model=injection_model,
                                                    parameters=settings.injection_parameters.__dict__,
                                                    waveform_arguments=settings.waveform_arguments.__dict__,
                                                    **settings.waveform_data.__dict__)

    hf_signal = waveform_generator.frequency_domain_strain()
    ifos = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
        name,
        injection_polarizations=hf_signal,
        injection_parameters=settings.injection_parameters.__dict__,
        outdir=outdir,
        zero_noise=settings.detector_settings.zero_noise,
        **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors]

    waveform_generator.time_domain_source_model = recovery_model

    likelihood = tupak.gw.likelihood \
        .GravitationalWaveTransient(interferometers=ifos,
                                    waveform_generator=waveform_generator,
                                    prior=settings.recovery_priors.proper_dict(),
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization)

    result = tupak.core.sampler.run_sampler(likelihood=likelihood,
                                            priors=settings.recovery_priors.proper_dict(),
                                            injection_parameters=settings.injection_parameters.__dict__,
                                            outdir=outdir,
                                            save=False,
                                            **settings.sampler_settings.__dict__)
    result.plot_corner(lionize=settings.other_settings.lionize)
    result.memory_settings = settings
    for key in settings:
        result['key'] = settings[key]
    # result.ifos = ifos
    print(result)

    result.save_to_file()

    super_dir = outdir.split("/")[0]
    filename = super_dir + '/distance_evidence.dat'
    with open(filename, 'a') as outfile:
        outfile.write(str(settings.injection_parameters.luminosity_distance) + '\t' +
                      str(result.log_bayes_factor) + '\t' +
                      str(result.log_evidence_err) + '\n')


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
    # priors['prior_total_mass'] = tupak.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
    # priors['prior_mass_ratio'] = tupak.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$")
    # priors['prior_luminosity_distance'] = tupak.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
    #                                                                            maximum=5e3, latex_label="$L_D$")
    # priors['prior_inc'] = tupak.core.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$")
    priors['prior_phase'] = tupak.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, latex_label="$\phi$")
    # priors['prior_ra'] = tupak.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, latex_label="$RA$")
    # priors['prior_dec'] = tupak.core.prior.Cosine(name='dec', latex_label="$DEC$")
    priors['prior_psi'] = tupak.core.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi, latex_label="$\psi$")
    # priors['prior_geocent_time'] = tupak.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time')
    nr_sur_kwargs = dict(
        start_time=start_time,
        duration=duration,
        label='NRSur',
        new_seed=True,
        zero_noise=False
    )
    nr_sur_kwargs.update(priors)
    nr_sur_kwargs.update(kwargs)
    run_basic_injection(injection_model=injection_model, recovery_model=recovery_model, outdir=outdir, **nr_sur_kwargs)


def run_basic_injection_imr_phenom(injection_model, recovery_model, outdir, **kwargs):
    priors = dict()
    injection_parameters = InjectionParameters.init_with_updated_kwargs(**kwargs)
    for key in injection_parameters.__dict__:
        priors['prior_' + key] = injection_parameters.__dict__[key]
    priors['prior_total_mass'] = tupak.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
    # priors['prior_mass_ratio'] = tupak.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$")
    # priors['prior_luminosity_distance'] = tupak.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
    #                                                                            maximum=5e3, latex_label="$L_D$")
    # priors['prior_inc'] = tupak.core.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$")
    # priors['prior_phase'] = tupak.core.prior.Uniform(name='phase', minimum=0, maximum=np.pi, latex_label="$\phi$")
    # priors['prior_ra'] = tupak.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, latex_label="$RA$")
    # priors['prior_dec'] = tupak.core.prior.Cosine(name='dec', latex_label="$DEC$")
    # priors['prior_psi'] = tupak.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, latex_label="$\psi$")
    # priors['prior_geocent_time'] = tupak.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time')

    imr_phenom_kwargs = dict(
        label='IMRPhenomD',
        new_seed=False,
        zero_noise=True,
        npoints=500,
        # time_marginaliztaion=True,
        # phase_marginalization=True
    )
    imr_phenom_kwargs.update(priors)
    imr_phenom_kwargs.update(kwargs)
    run_basic_injection(injection_model=injection_model, recovery_model=recovery_model, outdir=outdir,
                        **imr_phenom_kwargs)
