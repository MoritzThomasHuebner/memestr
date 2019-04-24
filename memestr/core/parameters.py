import numpy as np
import bilby


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
            res = res + '\t' + key + "=" + str(self.__dict__[key]) + ", \n"

        res = res[:-3] + ")"
        return res

    @staticmethod
    def _set_bool_from_string(param):
        if param == 'True' or param is True:
            param = True
        else:
            param = False
        return param


class InjectionParameters(RunParameters):

    def __init__(self, mass_ratio=0.8, total_mass=60.0, s11=0.0, s12=0.0, s13=0.0, s21=0.0, s22=0.0, s23=0.0,
                 luminosity_distance=500., inc=np.pi / 2, phase=1.3, ra=1.54, dec=-0.7, psi=0.9,
                 geocent_time=1126259642.413, random_injection_parameters=False):
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
        self.random_injection_parameters = random_injection_parameters


class RecoveryPriors(RunParameters):

    def __init__(self,
                 total_mass=bilby.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$"),
                 mass_ratio=bilby.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$"),
                 luminosity_distance=bilby.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
                                                                          maximum=5e3, latex_label="$L_D$"),
                 inc=bilby.core.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$"),
                 phase=bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, latex_label="$\phi$"),
                 ra=bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, latex_label="$RA$"),
                 dec=bilby.core.prior.Cosine(name='dec', latex_label="$DEC$"),
                 psi=bilby.core.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi, latex_label="$\psi$"),
                 geocent_time=bilby.core.prior.Uniform(1126259642.322, 1126259642.522, name='geocent_time'),
                 s11=bilby.core.prior.DeltaFunction(name='s11', peak=0.0, latex_label='s11'),
                 s12=bilby.core.prior.DeltaFunction(name='s12', peak=0.0, latex_label='s12'),
                 s21=bilby.core.prior.DeltaFunction(name='s21', peak=0.0, latex_label='s21'),
                 s22=bilby.core.prior.DeltaFunction(name='s22', peak=0.0, latex_label='s22'),
                 s13=bilby.gw.prior.AlignedSpin(name='s13', a_prior=bilby.core.prior.Uniform(0, 0.5),
                                                latex_label='s13'),
                 s23=bilby.gw.prior.AlignedSpin(name='s23', a_prior=bilby.core.prior.Uniform(0, 0.5),
                                                latex_label='s23')):
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
        for key in [
            'prior_total_mass', 'prior_mass_ratio', 'prior_luminosity_distance', 'prior_inc', 'prior_phase', 'prior_ra',
            'prior_dec', 'prior_psi', 'prior_geocent_time', 'prior_s11', 'prior_s12', 'prior_s21', 'prior_s22',
            'prior_s13', 'prior_s23'
        ]:
            result[key[6:]] = getattr(self, key)
        return result


class WaveformArguments(RunParameters):

    def __init__(self, l_max=2, alpha=None):
        super(WaveformArguments, self).__init__()
        self.l_max = l_max
        self.alpha = alpha


class WaveformData(RunParameters):

    def __init__(self, start_time=0, duration=8, sampling_frequency=4096):
        super(WaveformData, self).__init__()
        self.start_time = start_time
        self.duration = duration
        self.sampling_frequency = sampling_frequency


class CpnestSettings(RunParameters):

    def __init__(self, sampler='cpnest', npoints=200, label='IMRPhenomD', clean=False,
                 dlogz=0.1, maxmcmc=100, nthreads=1, resume=True):
        super(CpnestSettings, self).__init__()
        self.sampler = sampler
        self.npoints = int(npoints)
        self.label = label
        self.clean = clean
        self.nthreads = nthreads
        self.dlogz = dlogz
        self.maxmcmc = int(maxmcmc)
        self.resume = resume

    @property
    def resume(self):
        return self._resume

    @resume.setter
    def resume(self, resume):
        self._resume = self._set_bool_from_string(resume)

    @property
    def clean(self):
        return self._clean

    @clean.setter
    def clean(self, clean):
        self._clean = self._set_bool_from_string(clean)


class DetectorSettings(RunParameters):

    def __init__(self, zero_noise=False, *detectors):
        super(DetectorSettings, self).__init__()
        self.zero_noise = zero_noise
        self.detectors = list(detectors)
        if len(self.detectors) == 0:
            self.detectors = ['H1', 'L1', 'V1']

    @property
    def zero_noise(self):
        return self._zero_noise

    @zero_noise.setter
    def zero_noise(self, zero_noise):
        self._zero_noise = self._set_bool_from_string(zero_noise)


class OtherSettings(RunParameters):

    def __init__(self, lionize=False, time_marginalization=False,
                 distance_marginalization=False, phase_marginalization=False, random_seed=0):
        super(OtherSettings, self).__init__()
        self.random_seed = random_seed
        self.lionize = lionize
        self.phase_marginalization = phase_marginalization
        self.time_marginalization = time_marginalization
        self.distance_marginalization = distance_marginalization

    @property
    def lionize(self):
        return self._lionize

    @lionize.setter
    def lionize(self, lionize):
        self._lionize = self._set_bool_from_string(lionize)

    @property
    def distance_marginalization(self):
        return self._distance_marginalization

    @distance_marginalization.setter
    def distance_marginalization(self, distance_marginalization):
        self._distance_marginalization = self._set_bool_from_string(distance_marginalization)

    @property
    def time_marginalization(self):
        return self._time_marginalization

    @time_marginalization.setter
    def time_marginalization(self, time_marginalization):
        self._time_marginalization = self._set_bool_from_string(time_marginalization)

    @property
    def phase_marginalization(self):
        return self._phase_marginalization

    @phase_marginalization.setter
    def phase_marginalization(self, phase_marginalization):
        self._phase_marginalization = self._set_bool_from_string(phase_marginalization)


class AllSettings(object):

    def __init__(self, injection_parameters=InjectionParameters(), recovery_priors=RecoveryPriors(),
                 waveform_arguments=WaveformArguments(), waveform_data=WaveformData(),
                 sampler_settings=CpnestSettings(), detector_settings=DetectorSettings(),
                 other_settings=OtherSettings()):
        self.injection_parameters = injection_parameters
        self.recovery_priors = recovery_priors
        self.waveform_arguments = waveform_arguments
        self.waveform_data = waveform_data
        self.sampler_settings = sampler_settings
        self.detector_settings = detector_settings
        self.other_settings = other_settings

    def __repr__(self):
        return 'AllSettings(\n' + repr(self.injection_parameters) + \
               ', \n' + repr(self.recovery_priors) + \
               ', \n' + repr(self.waveform_arguments) + \
               ', \n' + repr(self.waveform_data) + \
               ', \n' + repr(self.sampler_settings) + \
               ', \n' + repr(self.detector_settings) + \
               ', \n' + repr(self.other_settings) + '\n)'

    @classmethod
    def from_defaults_with_some_specified_kwargs(cls, **kwargs):
        return cls(injection_parameters=InjectionParameters.init_with_updated_kwargs(**kwargs),
                   recovery_priors=RecoveryPriors.init_with_updated_kwargs(**kwargs),
                   waveform_arguments=WaveformArguments.init_with_updated_kwargs(**kwargs),
                   waveform_data=WaveformData.init_with_updated_kwargs(**kwargs),
                   sampler_settings=CpnestSettings.init_with_updated_kwargs(**kwargs),
                   detector_settings=DetectorSettings.init_with_updated_kwargs(**kwargs),
                   other_settings=OtherSettings.init_with_updated_kwargs(**kwargs))
