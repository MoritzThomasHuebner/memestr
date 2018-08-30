from __future__ import division
import numpy as np
import tupak


class AllSettings(object):

    def __init__(self, injection_parameters, recovery_priors, waveform_arguments, waveform_data, sampler_settings,
                 detector_settings, other_settings):
        self.injection_parameters = injection_parameters
        self.recovery_priors = recovery_priors
        self.waveform_arguments = waveform_arguments
        self.waveform_data = waveform_data
        self.sampler_settings = sampler_settings
        self.detector_settings = detector_settings
        self.other_settings = other_settings

        self.__update_arguments_into_injection_parameters()

    def __update_arguments_into_injection_parameters(self):
        for key in self.waveform_arguments.__dict__:
            self.injection_parameters.__dict__[key] = self.waveform_arguments.__dict__[key]

    @classmethod
    def from_defaults_with_some_specified_kwargs(cls, **kwargs):
        return cls(injection_parameters=InjectionParameters.init_with_updated_kwargs(**kwargs),
                   recovery_priors=RecoveryPriors.init_with_updated_kwargs(**kwargs),
                   waveform_arguments=WaveformArguments.init_with_updated_kwargs(**kwargs),
                   waveform_data=WaveformData.init_with_updated_kwargs(**kwargs),
                   sampler_settings=SamplerSettings.init_with_updated_kwargs(**kwargs),
                   detector_settings=DetectorSettings.init_with_updated_kwargs(**kwargs),
                   other_settings=OtherSettings.init_with_updated_kwargs(**kwargs))


class RunParameters(object):

    def __init__(self):
        pass

    def update_args(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

    @classmethod
    def init_with_updated_kwargs(cls, **kwargs):
        res = cls()
        res.update_args(**kwargs)
        return res


class InjectionParameters(RunParameters):

    def __init__(self, mass_ratio=1.2, total_mass=60, s11=0, s12=0, s13=0, s21=0, s22=0, s23=0,
                 luminosity_distance=500., inc=np.pi / 2, phase=1.3, ra=1.54, dec=-0.7, psi=2.659,
                 geocent_time=1126259642.413):
        super(InjectionParameters, self).__init__()
        self.mass_ratio = mass_ratio
        self.total_mass = total_mass
        self.s11 = s11
        self.s12 = s12
        self.s13 = s13
        self.s21 = s21
        self.s22 = s22
        self.s23 = s23
        self.luminosity_distance = luminosity_distance
        self.inc = inc
        self.phase = phase
        self.ra = ra
        self.dec = dec
        self.psi = psi
        self.geocent_time = geocent_time


class RecoveryPriors(RunParameters):

    def __init__(self,
                 total_mass=tupak.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$"),
                 mass_ratio=tupak.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$"),
                 luminosity_distance=tupak.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
                                                                          maximum=5e3, latex_label="$L_D$"),
                 inc=tupak.core.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$"),
                 phase=tupak.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, latex_label="$\phi$"),
                 ra=tupak.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, latex_label="$RA$"),
                 dec=tupak.core.prior.Cosine(name='dec', latex_label="$DEC$"),
                 psi=tupak.core.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi, latex_label="$\psi$"),
                 geocent_time=tupak.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time'),
                 s11=tupak.core.prior.Uniform(name='s11', minimum=0, maximum=0.8, latex_label='s11'),
                 s12=tupak.core.prior.Uniform(name='s12', minimum=0, maximum=0.8, latex_label='s12'),
                 s21=tupak.core.prior.Uniform(name='s21', minimum=0, maximum=0.8, latex_label='s21'),
                 s22=tupak.core.prior.Uniform(name='s22', minimum=0, maximum=0.8, latex_label='s22'),
                 s13=tupak.core.prior.Uniform(name='s13', minimum=0, maximum=0.8, latex_label='s13'),
                 s23=tupak.core.prior.Uniform(name='s23', minimum=0, maximum=0.8, latex_label='s23')):
        super(RecoveryPriors, self).__init__()
        self.prior_total_mass = total_mass
        self.prior_mass_ratio = mass_ratio
        self.prior_luminosity_distance = luminosity_distance
        self.prior_inc = inc
        self.prior_phase = phase
        self.prior_ra = ra
        self.prior_dec = dec
        self.prior_psi = psi
        self.prior_geocent_time = geocent_time
        self.prior_s11 = s11
        self.prior_s12 = s12
        self.prior_s21 = s21
        self.prior_s22 = s22
        self.prior_s13 = s13
        self.prior_s23 = s23

    def proper_dict(self):
        result = dict()
        for key in self.__dict__:
            result[key[6:]] = self.__dict__[key]
        return result


class WaveformArguments(RunParameters):

    def __init__(self, l_max=2):
        super(WaveformArguments, self).__init__()
        self.LMax = l_max


class WaveformData(RunParameters):

    def __init__(self, start_time=0, duration=4, sampling_frequency=4096):
        super(WaveformData, self).__init__()
        self.start_time = start_time
        self.duration = duration
        self.sampling_frequency = sampling_frequency


class SamplerSettings(RunParameters):

    def __init__(self, sampler='pymultinest', npoints=6000, label='IMRPhenomD'):
        super(SamplerSettings, self).__init__()
        self.sampler = sampler
        self.npoints = npoints
        self.label = label


class DetectorSettings(RunParameters):

    def __init__(self, zero_noise=False, *detectors):
        super(DetectorSettings, self).__init__()
        self.zero_noise = zero_noise
        self.detectors = list(detectors)
        if len(self.detectors) == 0:
            self.detectors = ['H1', 'L1', 'V1']


class OtherSettings(RunParameters):

    def __init__(self, new_seed=True, lionize=False):
        super(OtherSettings, self).__init__()
        self.new_seed = new_seed
        self.lionize = lionize


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

    likelihood = tupak.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                waveform_generator=waveform_generator,
                                                                prior=settings.recovery_priors.proper_dict())

    result = tupak.core.sampler.run_sampler(likelihood=likelihood,
                                            priors=settings.recovery_priors.proper_dict(),
                                            injection_parameters=settings.injection_parameters.__dict__,
                                            outdir=outdir,
                                            **settings.sampler_settings.__dict__)
    result.plot_corner(lionize=settings.other_settings.lionize)
    print(result)

    filename = 'distance_evidence_' + outdir[4:] + '.dat'
    with open(filename, 'a') as outfile:
        outfile.write(str(settings.injection_parameters.luminosity_distance) + '\t' +
                      str(result.log_evidence) + '\t' +
                      str(result.log_evidence_err) + '\n')


def update_kwargs(default_kwargs, kwargs):
    new_kwargs = default_kwargs.copy()
    for key in list(set(default_kwargs.keys()).intersection(kwargs.keys())):
        new_kwargs[key] = kwargs[key]
    return new_kwargs


def run_basic_injection_nrsur(injection_model, recovery_model, outdir, **kwargs):
    start_time = -0.5
    end_time = 0.00  # 0.01   0.029
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
        new_seed=False,
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
    # priors['prior_phase'] = tupak.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, latex_label="$\phi$")
    # priors['prior_ra'] = tupak.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, latex_label="$RA$")
    # priors['prior_dec'] = tupak.core.prior.Cosine(name='dec', latex_label="$DEC$")
    # priors['prior_psi'] = tupak.core.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi, latex_label="$\psi$")
    # priors['prior_geocent_time'] = tupak.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time')

    imr_phenom_kwargs = dict(
        label='IMRPhenomD',
        new_seed=True,
        zero_noise=True,
        npoints=500
    )
    imr_phenom_kwargs.update(priors)
    imr_phenom_kwargs.update(kwargs)
    run_basic_injection(injection_model=injection_model, recovery_model=recovery_model, outdir=outdir,
                        **imr_phenom_kwargs)
