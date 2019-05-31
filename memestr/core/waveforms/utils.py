import copy

import gwmemory
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal.windows import tukey

from memestr.core.waveforms import roll_off, wrap_by_n_indices


def get_alpha(kwargs, times):
    if 'alpha' in kwargs:
        if kwargs['alpha'] is not None:
            return kwargs['alpha']
    return roll_off / (times[-1] - times[0])


def apply_window(waveform, times, kwargs):
    alpha = get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)

    for mode in waveform:
        waveform[mode] *= window
    return waveform


def wrap_at_maximum(waveform):
    max_index = np.argmax(np.abs(np.abs(waveform['plus']) + np.abs(waveform['cross'])))
    shift = len(waveform['plus']) - max_index
    waveform = wrap_by_n_indices(shift=shift, waveform=copy.deepcopy(waveform))
    return waveform, shift


def wrap_at_maximum_memory_generator(waveform, memory_generator):
    max_index = np.argmax(memory_generator.h_lm[(2, 2)])
    shift = len(memory_generator.h_lm[(2, 2)]) - max_index
    waveform = wrap_by_n_indices(shift=shift, waveform=copy.deepcopy(waveform))
    return waveform, shift


def wrap_by_n_indices(shift, waveform):
    for mode in waveform:
        waveform[mode] = np.roll(waveform[mode], shift=shift)
    return waveform


def wrap_by_time_shift(waveforms, time_shifts, time_per_index):
    index_shifts = np.round(time_shifts / time_per_index).astype(int)
    return np.roll(waveforms, shift=index_shifts)


def wrap_by_time_shift_continuous(times, waveform, time_shift):
    interpolants = dict()
    for mode in waveform:
        waveform_interpolants = CubicSpline(times, waveform[mode], extrapolate='periodic')
        new_times = times - time_shift
        interpolants[mode] = waveform_interpolants(new_times)
    return interpolants


def _get_new_times(time_shifts, times):
    shifted_times = []
    for time_shift in time_shifts:
        shifted_times.append(times - time_shift)
    return np.array(shifted_times)


def apply_time_shift_frequency_domain(waveform, frequency_array, duration, shift):
    for mode in waveform:
        waveform[mode] = waveform[mode] * np.exp(-2j * np.pi * (duration + shift) * frequency_array)
    return waveform


def nfft_vectorizable(time_domain_strain, sampling_frequency):
    frequency_domain_strain = dict()
    for mode in time_domain_strain:
        frequency_domain_strain[mode] = np.fft.rfft(time_domain_strain[mode])
        frequency_domain_strain[mode] /= sampling_frequency
    return frequency_domain_strain


wrap_at_maximum_vectorized = np.vectorize(pyfunc=wrap_at_maximum, excluded=['kwargs'], otypes=[dict])
wrap_by_n_indices_vectorized = np.vectorize(pyfunc=wrap_by_n_indices, excluded=['shift'], otypes=[dict])
apply_window_vectorized = np.vectorize(pyfunc=apply_window, excluded=['times', 'kwargs'], otypes=[dict])
wrap_by_time_shift_continuous_vectorized = np.vectorize(pyfunc=wrap_by_time_shift_continuous,
                                                        excluded=['times', 'time_shift'],
                                                        otypes=[np.ndarray])
_get_new_times_vectorized = np.vectorize(pyfunc=_get_new_times, excluded=['times'])
gamma_lmlm = gwmemory.angles.load_gamma()
roll_off = 0.2