from __future__ import print_function, division, absolute_import

from . import angles, constants, harmonics
import numpy as np
import lal # noqa
import lalsimulation as lalsim # noqa
from scipy.interpolate import interp1d
from .constants import cc, GG, Mpc, solar_mass
from .utils import zero_pad_time_series, combine_modes


class MemoryGenerator(object):

    def __init__(self, name, h_lm=None, times=None, distance=None):
        self.name = name
        self.h_lm = h_lm
        self.h_mem_lm = None
        self.times = times
        if self.h_lm is not None:
            try:
                self.zero_pad_h_lm()
            except KeyError:
                pass
        self.distance = distance
        self._modes = None

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
            _ = self.time_domain_oscillatory()
        lms = self.modes

        dhlm_dt = dict()
        for lm in lms:
            dhlm_dt[lm] = np.gradient(self.h_lm[lm], self.delta_t)

        dhlm_dt_sq = dict()
        for lm in lms:
            for lmp in lms:
                index = (lm, lmp)
                dhlm_dt_sq[index] = dhlm_dt[lm] * np.conjugate(dhlm_dt[lmp])

        if gamma_lmlm is None:
            gamma_lmlm = angles.load_gamma()

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
                    [dhlm_dt_sq[((l1, m1), (l2, m2))] * gamma_lmlm[delta_m]['{}{}{}{}'.format(l1, m1, l2, m2)][ii]
                     for (l1, m1), (l2, m2) in dhlm_dt_sq.keys() if m1 - m2 == int(delta_m)], axis=0)

        self.h_mem_lm = {lm: const * np.cumsum(dh_mem_dt_lm[lm]) * self.delta_t for lm in dh_mem_dt_lm}
        if inc is None or phase is None:
            return self.h_mem_lm, self.times
        else:
            return combine_modes(self.h_mem_lm, inc, phase), self.times

    def time_domain_oscillatory(self, **kwargs):
        pass

    def zero_pad_h_lm(self):
        for lm in self.h_lm:
            self.h_lm[lm] = zero_pad_time_series(self.times, self.h_lm[lm])


class HybridSurrogate(MemoryGenerator):

    _surrogate_loaded = False
    MASS_TO_TIME = 4.925491025543576e-06

    def __init__(self, q, total_mass=None, spin_1=None,
                 spin_2=None, distance=None, l_max=4, modes=None, times=None,
                 minimum_frequency=10, reference_frequency=50., units='mks'):
        """
        Initialise Surrogate MemoryGenerator
        Parameters
        ----------
        l_max: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        q: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in MPC.
        spin_1: array-like
            Spin vector of more massive black hole.
        spin_2: array-like
            Spin vector of less massive black hole.
        times: array-like
            Time array to evaluate the waveforms on, default is
            np.linspace(-900, 100, 10001).
        """
        if not self._surrogate_loaded:
            import gwsurrogate # noqa
            self.sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
            self._surrogate_loaded = True

        self.q = q
        self.MTot = total_mass
        self.chi_1 = spin_1
        self.chi_2 = spin_2
        self.minimum_frequency = minimum_frequency
        self.distance = distance
        self.LMax = l_max
        self.modes = modes
        self.reference_frequency = reference_frequency
        self.units = units

        if total_mass is None:
            self.h_to_geo = 1
            self.t_to_geo = 1
        else:
            self.h_to_geo = self.distance * Mpc / self.MTot /\
                solar_mass / GG * cc ** 2
            self.t_to_geo = 1 / self.MTot / solar_mass / GG * cc ** 3

        self.h_lm = None
        self.times = times

        if times is not None and self.units == 'dimensionless':
            times *= self.t_to_geo

        h_lm, times = self.time_domain_oscillatory(modes=self.modes, times=times)

        MemoryGenerator.__init__(self, h_lm=h_lm, times=times, distance=distance, name='HybridSurrogate')

    @property
    def duration(self):
        return self.times[-1] - self.times[0] + self.delta_t

    @property
    def epsilon(self):
        return 100 * self.MASS_TO_TIME * self.MTot

    def time_domain_oscillatory(self, times=None, modes=None, inc=None,
                                phase=None):
        """
        Get the mode decomposition of the surrogate waveform.
        Calculates a BBH waveform using the surrogate models of Field et al.
        (2014), Blackman et al. (2017)
        http://journals.aps.org/prx/references/10.1103/PhysRevX.4.031006,
        https://arxiv.org/abs/1705.07089
        See https://data.black-holes.org/surrogates/index.html for more
        information.
        Parameters
        ----------
        times: np.array, optional
            Time array on which to evaluate the waveform.
        modes: list, optional
            List of modes to try to generate.
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes
            will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic
            modes will be returned.
        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        times -= times[0]
        t_nr = np.arange(-self.duration / 1.3 + self.epsilon, self.epsilon, self.delta_t)

        if self.h_lm is None:

            h_lm = self.sur([self.q, self.chi_1, self.chi_2], times=t_nr, f_low=0, M=self.MTot,
                            dist_mpc=self.distance, units='mks', f_ref=self.reference_frequency)

            del h_lm[(5, 5)]
            old_keys = [(ll, mm) for ll, mm in h_lm.keys()]
            for ll, mm in old_keys:
                if mm > 0:
                    h_lm[(ll, -mm)] = (- 1)**ll * np.conj(h_lm[(ll, mm)])

            available_modes = set(h_lm.keys())

            if modes is None:
                modes = available_modes

            if not set(modes).issubset(available_modes):
                print('Requested {} unavailable modes'.format(
                    ' '.join(set(modes).difference(available_modes))))
                modes = list(set(modes).union(available_modes))
                print('Using modes {}'.format(' '.join(modes)))

            h_lm = {(ell, m): h_lm[ell, m] for ell, m in modes}
            self.h_lm = h_lm
        else:
            h_lm = self.h_lm
            times = self.times
        t_nr -= t_nr[0]
        for mode in h_lm.keys():
            if len(times) != len(h_lm[mode]):
                h_lm[mode] = interp1d(t_nr, h_lm[mode], bounds_error=False, fill_value=0.0)(times)

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, phase), times

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        if q < 1:
            q = 1 / q
        if q > 8:
            raise ValueError('Surrogate waveform not valid for q>8.')
        self._q = q

    @property
    def chi_1(self):
        return self._chi_1

    @chi_1.setter
    def chi_1(self, spin_1):
        if spin_1 is None:
            self._chi_1 = 0.0
        elif len(np.atleast_1d(spin_1)) == 3:
            self._chi_1 = spin_1[2]
        else:
            self._chi_1 = spin_1

    @property
    def chi_2(self):
        return self._chi_2

    @chi_2.setter
    def chi_2(self, spin_2):
        if spin_2 is None:
            self._chi_2 = 0.0
        elif len(np.atleast_1d(spin_2)) == 3:
            self._chi_2 = spin_2[2]
        else:
            self._chi_2 = spin_2


class BaseSurrogate(MemoryGenerator):

    def __init__(self, q, name='', m_tot=None, s1=None, s2=None, distance=None, l_max=4, max_q=2, times=None):

        MemoryGenerator.__init__(self, name=name, distance=distance)

        self.max_q = max_q
        self.q = q
        self.MTot = m_tot
        self.s1 = s1
        self.s2 = s2
        self.LMax = l_max
        self.times = times

    @property
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        if q < 1:
            q = 1 / q
        if q > self.max_q:
            print(f'WARNING: Surrogate waveform not tested for q>{self.max_q}.')
        self.__q = q

    @property
    def s1(self):
        return self.__s1

    @s1.setter
    def s1(self, s1):
        if s1 is None:
            self.__s1 = np.array([0., 0., 0.])
        else:
            self.__s1 = np.array(s1)

    @property
    def s2(self):
        return self.__s2

    @s2.setter
    def s2(self, s2):
        if s2 is None:
            self.__s2 = np.array([0., 0., 0.])
        else:
            self.__s2 = np.array(s2)

    @property
    def m1(self):
        return self.MTot / (1 + self.q)

    @property
    def m2(self):
        return self.m1 * self.q

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
    def h_to_geo(self):
        if self.MTot is None:
            return 1
        else:
            return self.distance * constants.Mpc / self.MTot / constants.solar_mass / constants.GG * constants.cc ** 2

    @property
    def t_to_geo(self):
        if self.MTot is None:
            return None
        else:
            return 1 / self.MTot / constants.solar_mass / constants.GG * constants.cc ** 3

    @property
    def geo_to_t(self):
        return 1 / self.t_to_geo

    @property
    def geometric_times(self):
        if self.times is not None:
            return self.times * self.t_to_geo
        else:
            return None


class NRSur7dq4(BaseSurrogate):

    AVAILABLE_MODES = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (3, -3), (3, -2),
                       (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (4, -4), (4, -3),
                       (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

    _surrogate_loaded = False
    surrogate = None

    """
    Memory generator for a numerical relativity surrogate.
    Attributes
    ----------
    modes: dict
        Spherical harmonic modes which we have knowledge of, default is ell<=4.
    h_lm: dict
        Spherical harmonic decomposed time-domain strain.
    times: array
        Array on which waveform is evaluated.
    q: float
        Binary mass ratio
    MTot: float, optional
        Total binary mass in solar units.
    distance: float, optional
        Distance to the binary in MPC.
    S1: array
        Spin vector of more massive black hole.
    S2: array
        Spin vector of less massive black hole.
    """

    def __init__(self, q, total_mass=None, s1=None, s2=None, distance=None, l_max=4, modes=None, times=None,
                 minimum_frequency=20., reference_frequency=20., units='mks'):
        """
        Initialise Surrogate MemoryGenerator
        Parameters
        ----------
        l_max: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        q: float
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

        self.minimum_frequency = minimum_frequency
        self.reference_frequency = reference_frequency
        self.units = units
        self.l_max = l_max
        self.h_lm = None
        super().__init__(q=q, name='NRSur7dq4', m_tot=total_mass, s1=s1, s2=s2,
                         distance=distance, l_max=l_max, max_q=4, times=times)
        self.h_lm = self.time_domain_oscillatory(modes=modes)
        self.zero_pad_h_lm()

    def time_domain_oscillatory(self, modes=None, inc=None, phase=None):
        if self.h_lm is None:
            if modes is None:
                modes = self.AVAILABLE_MODES
            lal_params = lal.CreateDict()
            data = lalsim.SimInspiralChooseTDModes(
                0.0, self.delta_t, self.m1_si, self.m2_si, self.s1[0], self.s1[1], self.s1[2],
                self.s2[0], self.s2[1], self.s2[2], self.minimum_frequency,
                self.reference_frequency, self.distance_si, lal_params, self.l_max, lalsim.NRSur7dq4)
            self.h_lm = {(ell, m): lalsim.SphHarmTimeSeriesGetMode(data, ell, m).data.data for ell, m in modes}
            self.zero_pad_h_lm()

        if inc is None or phase is None:
            return self.h_lm
        else:
            return combine_modes(self.h_lm, inc, phase)

    def time_domain_oscillatory_from_polarisations(self, inc, phase):
        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            self.m1_si, self.m2_si, self.s1[0], self.s1[1], self.s1[2], self.s2[0], self.s2[1], self.s2[2],
            self.distance_si, inc, phase, 0.0, 0.0, 0.0, self.delta_t, self.minimum_frequency,
            self.reference_frequency, lal.CreateDict(), lalsim.NRSur7dq4)
        hpc = dict(plus=hp.data.data, cross=hc.data.data)
        return {mode: zero_pad_time_series(times=self.times, mode=hpc[mode]) for mode in hpc}


class Approximant(MemoryGenerator):

    AVAILABLE_MODES = [(2, 2), (2, -2)]
    _f_min = 20.
    _f_ref = 20
    _theta = 0.0
    _phi = 0.0
    _long_asc_nodes = 0.0
    _eccentricity = 0.0
    _mean_per_ano = 0.0

    def __init__(self, name, q, m_tot=60, s1=np.array([0, 0, 0]), s2=np.array([0, 0, 0]), distance=400, times=None):
        """
        Initialise Surrogate MemoryGenerator
        
        Parameters
        ----------
        name: str
            File name to load.
        q: float
            Binary mass ratio
        m_tot: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in Mpc.
        s1: array_like
            Spin vector of more massive black hole.
        s2: array_like
            Spin vector of less massive black hole.
        times: array_like
            Time array to evaluate the waveforms on, default is time array from lalsimulation.
            FIXME
        """
        self.q = q
        self.MTot = m_tot
        self.__S1 = s1
        self.__S2 = s2
        self._check_prececssion()

        MemoryGenerator.__init__(self, name=name, times=times, distance=distance)

    @property
    def m1(self):
        return self.MTot / (1 + self.q)

    @property
    def m2(self):
        return self.m1 * self.q

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
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        if q > 1:
            q = 1 / q
        self.__q = q

    @property
    def h_to_geo(self):
        return self.distance_si / (self.m1_si + self.m2_si) / constants.GG * constants.cc ** 2

    @property
    def t_to_geo(self):
        return 1 / (self.m1_si + self.m2_si) / constants.GG * constants.cc ** 3

    @property
    def delta_t(self):
        return self.times[1] - self.times[0]

    @property
    def s1(self):
        return self.__S1

    @s1.setter
    def s1(self, s1):
        if s1 is None:
            self.__S1 = np.array([0., 0., 0.])
        else:
            self.__S1 = np.array(s1)
        self._check_prececssion()

    @property
    def s2(self):
        return self.__S2

    @s2.setter
    def s2(self, s2):
        if s2 is None:
            self.__S2 = np.array([0., 0., 0.])
        else:
            self.__S2 = np.array(s2)
        self._check_prececssion()

    @property
    def approximant(self):
        return lalsim.GetApproximantFromString(self.name)

    def _check_prececssion(self):
        if abs(self.__S1[0]) > 0 or abs(self.__S1[1]) > 0 or abs(self.__S2[0]) > 0 or abs(self.__S2[1]) > 0:
            print('WARNING: Approximant decomposition works only for non-precessing waveforms.')
            print('Setting spins to be aligned')
            self.__S1[0], self.__S1[1] = 0., 0.
            self.__S2[0], self.__S2[1] = 0., 0.
            print('New spins are: S1 = {}, S2 = {}'.format(self.__S1, self.__S2))
        else:
            self.__S1 = list(self.__S1)
            self.__S2 = list(self.__S2)

    def time_domain_oscillatory(self, modes=None, inc=None, phase=None):
        """
        Get the mode decomposition of the waveform approximant.

        Since the waveforms we consider only contain content about the ell=|m|=2 modes.
        We can therefore evaluate the waveform for a face-on system, where only the (2, 2) mode
        is non-zero.

        Parameters
        ----------
        modes: list, optional
            List of modes to try to generate.
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic modes will be returned.

        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        if self.h_lm is None:
            modes = modes or self.AVAILABLE_MODES

            if not set(modes).issubset(self.AVAILABLE_MODES):
                print(f"Requested {' '.join(set(modes).difference(self.AVAILABLE_MODES))} unavailable modes")
                modes = set(modes).union(self.AVAILABLE_MODES)
                modes = [str(m) for m in modes]
                print(f"Using modes {' '.join(modes)}")

            wf_dict = lal.CreateDict()

            h_plus, h_cross = lalsim.SimInspiralChooseTDWaveform(
                self.m1_si, self.m2_si, self.s1[0], self.s1[1], self.s1[2], self.s2[0], self.s2[1], self.s2[2],
                self.distance_si, self._theta, self._phi, self._long_asc_nodes, self._eccentricity, self._mean_per_ano,
                self.delta_t, self._f_min, self._f_ref, wf_dict, self.approximant)

            h = h_plus.data.data - 1j * h_cross.data.data
            h_22 = h / harmonics.sYlm(-2, 2, 2, self._theta, self._phi)

            self.h_lm = {(2, 2): h_22, (2, -2): np.conjugate(h_22)}
            self.zero_pad_h_lm()

        if inc is None or phase is None:
            return self.h_lm
        else:
            return combine_modes(h_lm=self.h_lm, inc=inc, phase=phase)


class PhenomXHM(Approximant):

    AVAILABLE_MODES = [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (3, 2), (3, -2), (4, 4), (4, -4)]
    _f_min = 20.
    _f_ref = 20.
    _long_asc_nodes = 0.0
    _eccentricity = 0.0
    _mean_per_ano = 0.0

    def __init__(self, q, m_tot=60, s1=np.array([0, 0, 0]), s2=np.array([0, 0, 0]), distance=400, times=None):
        name = "IMRPhenomXHM"
        super().__init__(name, q, m_tot, s1, s2, distance, times)

    def time_domain_oscillatory(self, modes=None, inc=None, phase=None):
        if self.h_lm is None:
            if modes is None:
                modes = self.AVAILABLE_MODES

            self.h_lm = dict()
            for mode in modes:
                h_lm, times = self.single_mode_from_choose_td(ell=mode[0], m=mode[1])
                self.h_lm[mode] = h_lm
            self.zero_pad_h_lm()

        if inc is None or phase is None:
            return self.h_lm
        else:
            return combine_modes(h_lm=self.h_lm, inc=inc, phase=phase)

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
            self._f_min, self._f_ref, lalparams, lalsim.IMRPhenomXHM)

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

    def __init__(self, q, m_tot=None, chi_1=0., chi_2=0., distance=None,
                 max_q=20, times=None, minimum_frequency=35., ecc=0):

        self.max_q = max_q
        self.q = q
        self.MTot = m_tot
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.times = times
        self.ecc = ecc
        self._s1 = None
        self._s2 = None

        self.minimum_frequency = minimum_frequency
        super().__init__(name='TEOBResumS', h_lm=None, times=times, distance=distance)

    @property
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        if q < 1:
            q = 1 / q
        if q > self.max_q:
            print(f'WARNING: Surrogate waveform not tested for q>{self.max_q}.')
        self.__q = q

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
        return self.MTot / (1 + self.q)

    @property
    def m2(self):
        return self.m1 * self.q

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

    def time_domain_oscillatory(self, modes=None, inc=None, phase=None):
        if self.h_lm is None:
            import EOBRun_module # noqa
            if modes is None:
                modes = [[2, 2]]
            ks = self.modes_to_k(modes)

            coalescing_angle = phase if phase is not None else 0.0
            inclination = inc if inc is not None else 0.0
            self.h_lm = dict()
            for mode, k in zip(modes, ks):
                parameters = {
                    'M': self.MTot,
                    'q': self.q,  # q > 1
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
            print(self.h_lm)
            self.zero_pad_h_lm()

        if inc is None or phase is None:
            return self.h_lm
        else:
            return combine_modes(self.h_lm, inc, phase)
