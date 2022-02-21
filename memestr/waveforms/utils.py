from copy import deepcopy

try:
    import gwmemory
except ModuleNotFoundError:
    gwmemory = None
import numpy as np
from scipy.signal.windows import tukey


roll_off = 0.2
gamma_lmlm = gwmemory.angles.load_gamma()


def get_alpha(kwargs, times):
    if 'alpha' in kwargs:
        if kwargs['alpha'] is not None:
            return kwargs['alpha']
    return roll_off / (times[-1] - times[0])


def apply_window(waveform, times, kwargs):
    alpha = get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)
    for mode in waveform.keys():
        waveform[mode] *= window
    return waveform


def wrap_at_maximum(waveform):
    max_index = np.argmax(np.abs(waveform['plus'] - 1j * waveform['cross']))
    shift = len(waveform['plus']) - max_index
    waveform = wrap_by_n_indices(shift=shift, waveform=deepcopy(waveform))
    return waveform, shift


def wrap_at_maximum_from_modes(waveform, memory_generator, inc, phase):
    hpc = gwmemory.waveforms.combine_modes(h_lm=memory_generator.h_lm, inc=inc, phase=phase)
    max_index = np.argmax(np.abs(hpc['plus'] - 1j * hpc['cross']))
    shift = len(waveform['plus']) - max_index
    waveform = wrap_by_n_indices(shift=shift, waveform=deepcopy(waveform))
    return waveform, shift


def wrap_by_n_indices(shift, waveform):
    for mode in deepcopy(waveform):
        waveform[mode] = np.roll(waveform[mode], shift=shift)
    return waveform


def apply_time_shift_frequency_domain(waveform, frequency_array, duration, shift):
    wf = deepcopy(waveform)
    for mode in wf:
        wf[mode] = wf[mode] * np.exp(-2j * np.pi * (duration + shift) * frequency_array)
    return wf


def nfft(time_domain_strain, sampling_frequency):
    frequency_domain_strain = dict()
    for mode in time_domain_strain:
        frequency_domain_strain[mode] = np.fft.rfft(time_domain_strain[mode])
        frequency_domain_strain[mode] /= sampling_frequency
    return frequency_domain_strain


def convert_to_frequency_domain(series, waveform, **kwargs):
    waveform = apply_window(waveform=waveform, times=series.time_array, kwargs=kwargs)
    _, shift = wrap_at_maximum(waveform=waveform)
    return nfft_and_time_shift(kwargs, series, shift, waveform)


def convert_to_frequency_domain_with_memory(series, waveform, reference_waveform, **kwargs):
    waveform = apply_window(waveform=waveform, times=series.time_array, kwargs=kwargs)
    _, shift = wrap_at_maximum(waveform=reference_waveform)
    return nfft_and_time_shift(kwargs, series, shift, waveform)


def nfft_and_time_shift(kwargs, series, shift, waveform):
    time_shift = kwargs.get('time_shift', 0.)
    time_shift += shift * (series.time_array[1] - series.time_array[0])
    waveform_fd = nfft(waveform, series.sampling_frequency)
    for mode in waveform:
        indexes = np.where(series.frequency_array < kwargs.get('minimum_frequency', 20))
        waveform_fd[mode][indexes] = 0
    waveform_fd = apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
                                                    duration=series.duration, shift=time_shift)
    return waveform_fd
