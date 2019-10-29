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


class WaveformArguments(RunParameters):

    def __init__(self, l_max=2, alpha=0.1):
        super(WaveformArguments, self).__init__()
        self.l_max = l_max
        self.alpha = alpha
        self.minimum_frequency = 20.


class WaveformData(RunParameters):

    def __init__(self, start_time=0, duration=16, sampling_frequency=2048):
        super(WaveformData, self).__init__()
        self.start_time = start_time
        self.duration = duration
        self.sampling_frequency = sampling_frequency


class SamplerSettings(RunParameters):

    def __init__(self, outdir='.', sampler='dynesty', npoints=200, walks=50, label='IMRPhenomD', clean=False,
                 dlogz=0.1, resume=True, plot=False):
        super(SamplerSettings, self).__init__()
        self.sampler = sampler
        self.npoints = int(npoints)
        self.walks = int(walks)
        self.label = label
        self.clean = clean
        self.dlogz = dlogz
        self.resume = resume
        self.plot = plot
        self.outdir = outdir

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

    def __init__(self, zero_noise=False, filename_base=None, *detectors):
        super(DetectorSettings, self).__init__()
        self.filename_base = filename_base
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

    def __init__(self, injection_parameters=InjectionParameters(),
                 waveform_arguments=WaveformArguments(), waveform_data=WaveformData(),
                 sampler_settings=SamplerSettings(), detector_settings=DetectorSettings(),
                 other_settings=OtherSettings()):
        self.injection_parameters = injection_parameters
        self.waveform_arguments = waveform_arguments
        self.waveform_data = waveform_data
        self.sampler_settings = sampler_settings
        self.detector_settings = detector_settings
        self.other_settings = other_settings

    def __repr__(self):
        return 'AllSettings(\n' + repr(self.injection_parameters) + \
               ', \n' + repr(self.waveform_arguments) + \
               ', \n' + repr(self.waveform_data) + \
               ', \n' + repr(self.sampler_settings) + \
               ', \n' + repr(self.detector_settings) + \
               ', \n' + repr(self.other_settings) + '\n)'

    @classmethod
    def from_defaults_with_some_specified_kwargs(cls, **kwargs):
        return cls(injection_parameters=InjectionParameters.init_with_updated_kwargs(**kwargs),
                   waveform_arguments=WaveformArguments.init_with_updated_kwargs(**kwargs),
                   waveform_data=WaveformData.init_with_updated_kwargs(**kwargs),
                   sampler_settings=SamplerSettings.init_with_updated_kwargs(**kwargs),
                   detector_settings=DetectorSettings.init_with_updated_kwargs(**kwargs),
                   other_settings=OtherSettings.init_with_updated_kwargs(**kwargs))
