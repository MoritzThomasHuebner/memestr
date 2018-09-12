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

    def __repr__(self):
        return 'AllSettings(' + repr(self.injection_parameters) + \
                       ', \n' + repr(self.recovery_priors) + \
                       ', \n' + repr(self.waveform_arguments) + \
                       ', \n' + repr(self.waveform_data) + \
                       ', \n' + repr(self.sampler_settings) + \
                       ', \n' + repr(self.detector_settings) +  \
                       ', \n' + repr(self.other_settings)

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

    def __repr__(self):
        res = self.__class__.__name__ + "("
        for key in self.__dict__:
            res = res + key + "=" + str(self.__dict__[key]) + ", "
        res = res + ")"
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

    def __init__(self, new_seed=True, lionize=False, time_marginalization=False,
                 distance_marginalization=False, phase_marginalization=False):
        super(OtherSettings, self).__init__()
        self.new_seed = new_seed
        self.lionize = lionize
        if time_marginalization == 'True':
            self.time_marginalization = True
        else:
            self.time_marginalization = False
        if distance_marginalization == 'True':
            self.distance_marginalization = True
        else:
            self.distance_marginalization = False
        if phase_marginalization == 'True':
            self.phase_marginalization = True
        else:
            self.phase_marginalization = False
