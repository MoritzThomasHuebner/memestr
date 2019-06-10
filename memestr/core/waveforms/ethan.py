import NRSur7dq2
import gwsurrogate as gws
import matplotlib
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
import gwmemory

params = {
    'axes.labelsize': 18,
    'font.size': 24,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.titlesize': 20,
    'text.usetex': True,
    'figure.figsize': [8, 6],
    'font.family': 'sans',
    'font.serif': 'DejaVu'
}
matplotlib.rcParams.update(params)

MASS_TO_TIME = 4.925491025543576e-06  # solar masses to seconds
MASS_TO_DISTANCE = 4.785415917274702e-20  # solar masses to Mpc
MPC = 3.08568e22  # Mpc in metres
MSUN = 1.98855e30  # solar mass in  kg

# Evaluate the NRHybSur waveform
sur = gws.LoadSurrogate('NRHybSur3dq8')


def convert_time_strain_to_frequency(h, t, time, sampling_frequency, minimum_frequency, frequency):
    h = interp1d(t, h, bounds_error=False, fill_value=0.0)(time)
    h = h * tukey(len(h), alpha=0.1)

    plus_t = np.real(h)
    cross_t = -np.imag(h)

    plus = np.fft.rfft(plus_t) / sampling_frequency
    cross = np.fft.rfft(cross_t) / sampling_frequency

    plus[frequency < minimum_frequency] = complex(0.0)
    cross[frequency < minimum_frequency] = complex(0.0)

    return plus, cross


def gws_nominal(frequency, mass_1, mass_2, luminosity_distance, chi_1,
                chi_2, theta_jn, phase, psi,
                geocent_time, ra, dec, **kwargs):
    waveform_kwargs = dict(reference_frequency=50.0, minimum_frequency=20.0)
    waveform_kwargs.update(kwargs)
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']

    duration = 1. / (frequency[1] - frequency[0])
    dt = 1. / (2 * frequency[-1])
    time = np.arange(0.0, duration, dt)
    sampling_frequency = frequency[-1] * 2.0

    spin_1x = 0.0
    spin_1y = 0.0
    spin_2x = 0.0
    spin_2y = 0.0
    spin_1z = chi_1
    spin_2z = chi_2
    q = mass_1 / mass_2
    m_tot = mass_1 + mass_2

    s1 = np.array([spin_1x, spin_1y, spin_1z])
    s2 = np.array([spin_2x, spin_2y, spin_2z])

    x = [q, s1[2], s2[2]]
    modes_full = [(2, 2), (2, 1), (2, 0), (3, 3), (3, 2),
                  (3, 1), (3, 0), (4, 4), (4, 3), (4, 2), (5, 5)]

    epsilon = 100 * MASS_TO_TIME * m_tot
    t_nr = np.arange(-duration / 1.3 + epsilon, epsilon, dt)

    h = sur(x, times=t_nr, f_low=0, M=m_tot,
            dist_mpc=luminosity_distance, units='mks', f_ref=reference_frequency)
    t_nr -= t_nr[0]

    h_nr = np.zeros(len(h[modes_full[0]]), dtype=complex)
    y22_time_shift = 0
    for mode in modes_full:
        h_nr += (gwmemory.harmonics.sYlm(-2, mode[0], mode[1], theta_jn, phase + np.pi / 2) * h[mode])

        if mode[1] > 0:
            h_nr += (gwmemory.harmonics.sYlm(-2, mode[0], -mode[1], theta_jn, phase + np.pi / 2) *
                     (-1) ** mode[0] * np.conj(h[mode]))

        if mode == (2, 2):
            y22_time_shift = t_nr[np.argmax(h[(2, 2)])]

    plus, cross = convert_time_strain_to_frequency(h_nr, t_nr, time,
                                                   sampling_frequency, minimum_frequency, frequency)

    plus = plus * np.exp(-2j * np.pi * (duration - y22_time_shift) * frequency)
    cross = cross * np.exp(-2j * np.pi * (duration - y22_time_shift) * frequency)

    return {'plus': plus, 'cross': cross}
