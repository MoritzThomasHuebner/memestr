from __future__ import print_function, division, absolute_import

from itertools import product
import lal # noqa
import lalsimulation as lalsim # noqa
import numpy as np
from scipy.interpolate import interp1d

from . import angles, constants, harmonics
from .constants import cc, GG, Mpc, solar_mass
from .utils import zero_pad_time_series, combine_modes


class MemoryGenerator(object):

    def __init__(self, times=None, distance=None, modes=None):
        self.h_lm = None
        self.h_mem_lm = None
        self.times = times
        self.distance = distance
        self.modes = modes

    @property
    def modes(self):
        if self._modes is None and self.h_lm is not None:
            self._modes = self.h_lm.keys()
        return self._modes

    @modes.setter
    def modes(self, modes):
        self._modes = modes

    @property
    def delta_t(self):
        return self.times[1] - self.times[0]

    @property
    def duration(self):
        return self.times[-1] - self.times[0]

    @property
    def sampling_frequency(self):
        return 1/(self.times[1] - self.times[0])

    @property
    def delta_f(self):
        return 1/self.duration

    def time_domain_memory(self, inc, phase, gamma_lmlm=None):
        """
        Calculate the spherical harmonic decomposition of the nonlinear memory from a dictionary of spherical mode time
        series

        Parameters
        ----------
        inc: float
            Inclination of the source, if None, the spherical harmonic modes will be returned.
        phase: float
            Reference phase of the source, if None, the spherical harmonic modes will be returned.
            For CBCs this is the phase at coalescence.
        gamma_lmlm: dict
            Dictionary of arrays defining the angular dependence of the different memory modes, default=None
            if None the function will attempt to load them

        Return
        ------
        h_mem_lm: dict
            Time series of the spherical harmonic decomposed memory waveform.
        times: array_like
            Time series on which memory is evaluated.
        """
        if self.h_lm is None:
            self.set_h_lm()

        dhlm_dt = {lm: np.gradient(self.h_lm[lm], self.delta_t) for lm in self.modes}
        dhlm_dt_sq = {(lm, lmp): dhlm_dt[lm] * np.conjugate(dhlm_dt[lmp])
                      for lm, lmp in product(self.modes, self.modes)}
        gamma_lmlm = gamma_lmlm or angles.load_gamma()

        # constant terms in SI units
        const = 1 / 4 / np.pi
        if self.distance is not None:
            const *= self.distance * constants.Mpc / constants.cc

        dh_mem_dt_lm = dict()
        for ii, ell in enumerate(gamma_lmlm['0'].l):
            if ell > 4:
                continue
            for delta_m in gamma_lmlm.keys():
                if abs(int(delta_m)) > ell:
                    continue
                dh_mem_dt_lm[(ell, int(delta_m))] = np.sum(
                    [dhlm_dt_sq[((l1, m1), (l2, m2))] * gamma_lmlm[delta_m][f"{l1}{m1}{l2}{m2}"][ii]
                     for (l1, m1), (l2, m2) in dhlm_dt_sq.keys() if m1 - m2 == int(delta_m)], axis=0)

        self.h_mem_lm = {lm: const * np.cumsum(dh_mem_dt_lm[lm]) * self.delta_t for lm in dh_mem_dt_lm}
        if inc is None or phase is None:
            return self.h_mem_lm
        else:
            return combine_modes(self.h_mem_lm, inc, phase)

    def time_domain_oscillatory(self, inc, phase):
        """ Parameters
        ----------
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes
            will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic
            modes will be returned.
        """
        return combine_modes(self.h_lm, inc, phase)

    def set_h_lm(self):
        pass

    def zero_pad_h_lm(self):
        for lm in self.h_lm:
            self.h_lm[lm] = zero_pad_time_series(self.times, self.h_lm[lm])


class HybridSurrogate(MemoryGenerator):

    _surrogate_loaded = False
    MASS_TO_TIME = 4.925491025543576e-06

    def __init__(self, mass_ratio, total_mass=None, s1=None,
                 s2=None, distance=None, l_max=4, modes=None, times=None,
                 minimum_frequency=10, reference_frequency=50.):
        """
        Initialise Surrogate MemoryGenerator
        Parameters
        ----------
        l_max: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        mass_ratio: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in MPC.
        s1: array-like
            Spin vector of more massive black hole.
        s2: array-like
            Spin vector of less massive black hole.
        times: array-like
            Time array to evaluate the waveforms on, default is
            np.linspace(-900, 100, 10001).
        """
        if not self._surrogate_loaded:
            import gwsurrogate # noqa
            self.sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
            self._surrogate_loaded = True

        MemoryGenerator.__init__(self, times=times, distance=distance, modes=modes)

        self.mass_ratio = mass_ratio
        self.total_mass = total_mass
        self.s1 = s1
        self.s2 = s2
        self.minimum_frequency = minimum_frequency
        self.distance = distance
        self.LMax = l_max
        self.reference_frequency = reference_frequency

        self.times = times
        self.t_nr = np.arange(-self.duration / 1.3 + self.epsilon, self.epsilon, self.delta_t)
        self.set_h_lm()

    @property
    def duration(self):
        return self.times[-1] - self.times[0] + self.delta_t

    @property
    def epsilon(self):
        return 100 * self.MASS_TO_TIME * self.total_mass

    def set_h_lm(self):
        """
        Get the mode decomposition of the surrogate waveform.
        Calculates a BBH waveform using the surrogate models of Field et al.
        (2014), Blackman et al. (2017)
        http://journals.aps.org/prx/references/10.1103/PhysRevX.4.031006,
        https://arxiv.org/abs/1705.07089
        See https://data.black-holes.org/surrogates/index.html for more
        information.
        """
        if self.h_lm is None:
            h_lm = self.sur([self.mass_ratio, self.s1, self.s2], times=self.t_nr, f_low=0, M=self.total_mass,
                            dist_mpc=self.distance, units='mks', f_ref=self.reference_frequency)

            del h_lm[(5, 5)]
            old_keys = [(ll, mm) for ll, mm in h_lm.keys()]
            for ll, mm in old_keys:
                if mm > 0:
                    h_lm[(ll, -mm)] = (- 1) ** ll * np.conj(h_lm[(ll, mm)])

            available_modes = set(h_lm.keys())
            modes = self.modes or available_modes

            if not set(modes).issubset(available_modes):
                print('Requested {} unavailable modes'.format(
                    ' '.join(set(modes).difference(available_modes))))
                modes = list(set(modes).union(available_modes))
                print('Using modes {}'.format(' '.join(modes)))

            self.h_lm = {(ell, m): h_lm[ell, m] for ell, m in modes}
            times = self.times - self.times[0]
            t_nr = np.arange(-self.duration / 1.3 + self.epsilon, self.epsilon, self.delta_t)
            t_nr -= self.t_nr - self.t_nr[0]
            for mode in self.h_lm.keys():
                if len(times) != len(self.h_lm[mode]):
                    self.h_lm[mode] = interp1d(t_nr, self.h_lm[mode], bounds_error=False, fill_value=0.0)(times)

    @property
    def mass_ratio(self):
        return self._mass_ratio

    @mass_ratio.setter
    def mass_ratio(self, mass_ratio):
        if mass_ratio < 1:
            mass_ratio = 1 / mass_ratio
        if mass_ratio > 8:
            raise ValueError('Surrogate waveform not valid for q>8.')
        self._mass_ratio = mass_ratio

    @property
    def s1(self):
        return self._s1

    @s1.setter
    def s1(self, s1):
        if s1 is None:
            self._s1 = 0.0
        elif len(np.atleast_1d(s1)) == 3:
            self._s1 = s1[2]
        else:
            self._s1 = s1

    @property
    def s2(self):
        return self._s2

    @s2.setter
    def s2(self, s2):
        if s2 is None:
            self._s2 = 0.0
        elif len(np.atleast_1d(s2)) == 3:
            self._s2 = s2[2]
        else:
            self._s2 = s2


class BaseSurrogate(MemoryGenerator):

    MAX_Q = 2

    def __init__(
            self, mass_ratio, total_mass=None, s1=None, s2=None,
            distance=None, l_max=4, times=None, modes=None):

        MemoryGenerator.__init__(self, distance=distance, modes=modes)

        self.mass_ratio = mass_ratio
        self.total_mass = total_mass
        self.s1 = s1
        self.s2 = s2
        self.l_max = l_max
        self.times = times

    @property
    def mass_ratio(self):
        return self._mass_ratio

    @mass_ratio.setter
    def mass_ratio(self, mass_ratio):
        if mass_ratio < 1:
            mass_ratio = 1 / mass_ratio
        if mass_ratio > self.MAX_Q:
            print(f'WARNING: Surrogate waveform not tested for q>{self.MAX_Q}.')
        self._mass_ratio = mass_ratio

    @property
    def s1(self):
        return self._s1

    @s1.setter
    def s1(self, s1):
        if s1 is None:
            self._s1 = np.array([0., 0., 0.])
        else:
            self._s1 = np.array(s1)

    @property
    def s2(self):
        return self._s2

    @s2.setter
    def s2(self, s2):
        if s2 is None:
            self._s2 = np.array([0., 0., 0.])
        else:
            self._s2 = np.array(s2)

    @property
    def m1(self):
        return self.total_mass / (1 + self.mass_ratio)

    @property
    def m2(self):
        return self.m1 * self.mass_ratio

    @property
    def m1_si(self):
        return self.m1 * constants.solar_mass

    @property
    def m2_si(self):
        return self.m2 * constants.solar_mass

    @property
    def distance_si(self):
        return self.distance * constants.Mpc


class NRSur7dq4(BaseSurrogate):

    AVAILABLE_MODES = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (3, -3), (3, -2),
                       (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (4, -4), (4, -3),
                       (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

    _surrogate_loaded = False
    surrogate = None

    MAX_Q = 6

    def __init__(self, mass_ratio, total_mass=None, s1=None, s2=None, distance=None, l_max=4, modes=None, times=None,
                 minimum_frequency=20., reference_frequency=20.):
        """
        Initialise Surrogate MemoryGenerator
        Parameters
        ----------
        l_max: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        mass_ratio: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in MPC.
        s1: array-like
            Spin vector of more massive black hole.
        s2: array-like
            Spin vector of less massive black hole.
        times: array-like
            Time array to evaluate the waveforms on, default is
            np.linspace(-900, 100, 10001).
        """
        if not self._surrogate_loaded:
            import gwsurrogate # noqa
            self.surrogate = gwsurrogate.LoadSurrogate('NRSur7dq4')
            self._surrogate_loaded = True

        super().__init__(mass_ratio=mass_ratio, total_mass=total_mass, s1=s1, s2=s2,
                         distance=distance, l_max=l_max, times=times, modes=modes)

        self.minimum_frequency = minimum_frequency
        self.reference_frequency = reference_frequency
        self.l_max = l_max
        self.set_h_lm()

    def set_h_lm(self):
        if self.h_lm is None:
            modes = self.modes or self.AVAILABLE_MODES
            data = lalsim.SimInspiralChooseTDModes(
                0.0, self.delta_t, self.m1_si, self.m2_si, self.s1[0], self.s1[1], self.s1[2],
                self.s2[0], self.s2[1], self.s2[2], self.minimum_frequency,
                self.reference_frequency, self.distance_si, lal.CreateDict(), self.l_max, lalsim.NRSur7dq4)
            self.h_lm = {(ell, m): lalsim.SphHarmTimeSeriesGetMode(data, ell, m).data.data for ell, m in modes}
            self.zero_pad_h_lm()

    def time_domain_oscillatory_from_polarisations(self, inc, phase):
        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            self.m1_si, self.m2_si, self.s1[0], self.s1[1], self.s1[2], self.s2[0], self.s2[1], self.s2[2],
            self.distance_si, inc, phase, 0.0, 0.0, 0.0, self.delta_t, self.minimum_frequency,
            self.reference_frequency, lal.CreateDict(), lalsim.NRSur7dq4)
        hpc = dict(plus=hp.data.data, cross=hc.data.data)
        return {mode: zero_pad_time_series(times=self.times, mode=hpc[mode]) for mode in hpc}


class Approximant(MemoryGenerator):

    AVAILABLE_MODES = [(2, 2), (2, -2)]
    minimum_frequency = 20.
    reference_frequency = 20
    _theta = 0.0
    _phi = 0.0
    _long_asc_nodes = 0.0
    _eccentricity = 0.0
    _mean_per_ano = 0.0

    def __init__(
            self, name, mass_ratio, total_mass=60, s1=np.array([0, 0, 0]),
            s2=np.array([0, 0, 0]), distance=400, times=None, modes=None):
        """
        Initialise Surrogate MemoryGenerator
        
        Parameters
        ----------
        name: str
            File name to load.
        mass_ratio: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in Mpc.
        s1: array_like
            Spin vector of more massive black hole.
        s2: array_like
            Spin vector of less massive black hole.
        times: array_like
            Time array to evaluate the waveforms on, default is time array from lalsimulation.
        """
        MemoryGenerator.__init__(self, times=times, distance=distance, modes=modes)
        self.name = name
        self.mass_ratio = mass_ratio
        self.total_mass = total_mass
        self._s1 = s1
        self._s2 = s2
        self._check_prececssion()
        self.set_h_lm()

    @property
    def m1(self):
        return self.total_mass / (1 + self.mass_ratio)

    @property
    def m2(self):
        return self.m1 * self.mass_ratio

    @property
    def m1_si(self):
        return self.m1 * constants.solar_mass

    @property
    def m2_si(self):
        return self.m2 * constants.solar_mass

    @property
    def distance_si(self):
        return self.distance * constants.Mpc

    @property
    def mass_ratio(self):
        return self._mass_ratio

    @mass_ratio.setter
    def mass_ratio(self, mass_ratio):
        if mass_ratio > 1:
            mass_ratio = 1 / mass_ratio
        self._mass_ratio = mass_ratio

    @property
    def s1(self):
        return self._s1

    @s1.setter
    def s1(self, s1):
        if s1 is None:
            self._s1 = np.array([0., 0., 0.])
        else:
            self._s1 = np.array(s1)
        self._check_prececssion()

    @property
    def s2(self):
        return self._s2

    @s2.setter
    def s2(self, s2):
        if s2 is None:
            self._s2 = np.array([0., 0., 0.])
        else:
            self._s2 = np.array(s2)
        self._check_prececssion()

    @property
    def approximant(self):
        return lalsim.GetApproximantFromString(self.name)

    def _check_prececssion(self):
        if abs(self._s1[0]) > 0 or abs(self._s1[1]) > 0 or abs(self._s2[0]) > 0 or abs(self._s2[1]) > 0:
            print('WARNING: Approximant decomposition works only for non-precessing waveforms.')
            print('Setting spins to be aligned')
            self._s1[0], self._s1[1] = 0., 0.
            self._s2[0], self._s2[1] = 0., 0.
            print('New spins are: S1 = {}, S2 = {}'.format(self._s1, self._s2))
        else:
            self._s1 = list(self._s1)
            self._s2 = list(self._s2)

    def set_h_lm(self, modes=None):
        """
        Get the mode decomposition of the waveform approximant.

        Since the waveforms we consider only contain content about the ell=|m|=2 modes.
        We can therefore evaluate the waveform for a face-on system, where only the (2, 2) mode
        is non-zero.

        Parameters
        ----------
        modes: list, optional
            List of modes to try to generate.
        """
        if self.h_lm is None:
            modes = self.modes or self.AVAILABLE_MODES

            if not set(modes).issubset(self.AVAILABLE_MODES):
                print(f"Requested {' '.join(set(modes).difference(self.AVAILABLE_MODES))} unavailable modes")
                modes = set(modes).union(self.AVAILABLE_MODES)
                modes = [str(m) for m in modes]
                print(f"Using modes {' '.join(modes)}")

            wf_dict = lal.CreateDict()

            h_plus, h_cross = lalsim.SimInspiralChooseTDWaveform(
                self.m1_si, self.m2_si, self.s1[0], self.s1[1], self.s1[2], self.s2[0], self.s2[1], self.s2[2],
                self.distance_si, self._theta, self._phi, self._long_asc_nodes, self._eccentricity, self._mean_per_ano,
                self.delta_t, self.minimum_frequency, self.reference_frequency, wf_dict, self.approximant)

            h = h_plus.data.data - 1j * h_cross.data.data
            h_22 = h / harmonics.sYlm(-2, 2, 2, self._theta, self._phi)

            self.h_lm = {(2, 2): h_22, (2, -2): np.conjugate(h_22)}
            self.zero_pad_h_lm()


class PhenomXHM(Approximant):

    AVAILABLE_MODES = [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (3, 2), (3, -2), (4, 4), (4, -4)]

    def __init__(
            self, mass_ratio, total_mass=60, s1=np.array([0, 0, 0]), s2=np.array([0, 0, 0]),
            distance=400, times=None, modes=None
    ):
        super().__init__(
            name="IMRPhenomXHM", mass_ratio=mass_ratio, total_mass=total_mass,
            s1=s1, s2=s2, distance=distance, times=times, modes=modes)
        self.set_h_lm()

    def set_h_lm(self, modes=None):
        if self.h_lm is None:
            modes = self.modes or self.AVAILABLE_MODES
            self.h_lm = {mode: self.single_mode_from_choose_td(ell=mode[0], m=mode[1])[0] for mode in modes}
            self.zero_pad_h_lm()

    def time_domain_oscillatory_from_polarisations(self, inc, phase):
        lalparams = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalparams, 0)
        hpc, _ = self.get_polarisations(inc=inc, phase=phase, lalparams=lalparams)
        return {mode: zero_pad_time_series(times=self.times, mode=hpc[mode]) for mode in hpc}

    def single_mode_from_choose_td(self, ell, m):
        inc = 0.4
        phi = np.pi / 2
        lalparams = self._get_single_mode_lalparams_dict(ell, m)

        hpc, times = self.get_polarisations(inc=inc, phase=phi, lalparams=lalparams)
        hlm = (hpc['plus'] - 1j * hpc['cross']) / lal.SpinWeightedSphericalHarmonic(inc, np.pi - phi, -2, ell, m)
        return hlm, times

    def get_polarisations(self, inc, phase, lalparams):
        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            self.m1_si, self.m2_si, self.s1[0], self.s1[1], self.s1[2], self.s2[0], self.s2[1], self.s2[2],
            self.distance_si, inc, phase, self._long_asc_nodes, self._eccentricity, self._mean_per_ano, self.delta_t,
            self.minimum_frequency, self.reference_frequency, lalparams, lalsim.IMRPhenomXHM)

        shift = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds / 1e9
        times = np.arange(len(hp.data.data)) * self.delta_t + shift
        hpc = dict(plus=hp.data.data, cross=hc.data.data)
        return hpc, times

    @staticmethod
    def _get_single_mode_lalparams_dict(ell, m):
        lalparams = lal.CreateDict()
        mode_array = lalsim.SimInspiralCreateModeArray()
        lalsim.SimInspiralModeArrayActivateMode(mode_array, ell, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, mode_array)
        lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalparams, 0)
        return lalparams


class TEOBResumS(MemoryGenerator):

    AVAILABLE_MODES = None
    MAX_Q = 20

    def __init__(self, mass_ratio, total_mass=None, chi_1=0., chi_2=0., distance=None,
                 times=None, minimum_frequency=35., ecc=0, modes=None):

        super().__init__(times=times, distance=distance, modes=modes)
        self.mass_ratio = mass_ratio
        self.total_mass = total_mass
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.times = times
        self.ecc = ecc
        self._s1 = None
        self._s2 = None

        self.minimum_frequency = minimum_frequency
        self.set_h_lm()

    @property
    def mass_ratio(self):
        return self._mass_ratio

    @mass_ratio.setter
    def mass_ratio(self, mass_ratio):
        if mass_ratio < 1:
            mass_ratio = 1 / mass_ratio
        if mass_ratio > self.MAX_Q:
            print(f'WARNING: Waveform not tested for q>{self.MAX_Q}.')
        self._mass_ratio = mass_ratio

    @property
    def s1(self):
        return self._s1

    @s1.setter
    def s1(self, s1):
        if s1 is None:
            self._s1 = np.array([0., 0., 0.])
        else:
            self._s1 = np.array(s1)

    @property
    def s2(self):
        return self._s2

    @s2.setter
    def s2(self, s2):
        if s2 is None:
            self._s2 = np.array([0., 0., 0.])
        else:
            self._s2 = np.array(s2)

    @property
    def m1(self):
        return self.total_mass / (1 + self.mass_ratio)

    @property
    def m2(self):
        return self.m1 * self.mass_ratio

    @property
    def m1_si(self):
        return self.m1 * constants.solar_mass

    @property
    def m2_si(self):
        return self.m2 * constants.solar_mass

    @property
    def distance_si(self):
        return self.distance * constants.Mpc

    @staticmethod
    def modes_to_k(ms):
        return [int(x[0] * (x[0] - 1) / 2 + x[1] - 2) for x in ms]

    def set_h_lm(self):
        if self.h_lm is None:
            import EOBRun_module  # noqa
            modes = self.modes or [[2, 2]]
            ks = self.modes_to_k(modes)

            coalescing_angle = 0.0
            inclination = 0.0
            self.h_lm = dict()
            for mode, k in zip(modes, ks):
                parameters = {
                    'M': self.total_mass,
                    'q': self.mass_ratio,  # q > 1
                    'ecc': self.ecc,
                    'Lambda1': 0.,
                    'Lambda2': 0.,
                    'chi1': self.chi_1,
                    'chi2': self.chi_2,
                    'coalescence_angle': coalescing_angle,
                    'domain': 0,  # TD
                    'arg_out': 1,  # Output hlm/hflm. Default = 0
                    'use_mode_lm': [k],  # List of modes to use/output through EOBRunPy
                    'srate_interp': 4096.,  # srate at which to interpolate. Default = 4096.
                    'use_geometric_units': 0,  # Output quantities in geometric units. Default = 1
                    'initial_frequency': self.minimum_frequency,
                    # in Hz if use_geometric_units = 0, else in geometric units
                    'interp_uniform_grid': 1,
                    # Interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
                    'distance': self.distance,
                    'inclination': inclination,
                    # - (np.pi / 4), # = iota for non-precessing; adjusted to match IMRPhenomD definition
                    'output_hpc': 0
                }

                t, h_plus, h_cross, hlm, dyn = EOBRun_module.EOBRunPy(parameters)

                h = h_plus - 1j * h_cross
                h_lm = h / harmonics.sYlm(-2, mode[0], mode[1], inclination, coalescing_angle)

                self.h_lm.update({(mode[0], mode[1]): h_lm, (mode[0], -mode[1]): np.conjugate(h_lm)})
            self.zero_pad_h_lm()
