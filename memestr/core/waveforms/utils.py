from copy import deepcopy

import gwmemory
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
    for mode in deepcopy(waveform):
        waveform[mode] *= window
    return waveform


def wrap_at_maximum(waveform):
    max_index = np.argmax(np.abs(np.abs(waveform['plus']) + np.abs(waveform['cross'])))
    shift = len(waveform['plus']) - max_index
    waveform = wrap_by_n_indices(shift=shift, waveform=deepcopy(waveform))
    return waveform, shift


def wrap_at_maximum_from_2_2_mode(waveform, memory_generator):
    try:
        max_index = np.argmax(memory_generator.h_lm[(2, 2)])
        shift = len(memory_generator.h_lm[(2, 2)]) - max_index
    except KeyError:
        ref_mode = list(memory_generator.h_lm.keys())[0]
        max_index = np.argmax(memory_generator.h_lm[ref_mode])
        shift = len(memory_generator.h_lm[ref_mode]) - max_index
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


def convert_to_frequency_domain(memory_generator, series, waveform, **kwargs):
    waveform = apply_window(waveform=waveform, times=series.time_array, kwargs=kwargs)
    # _, shift = wrap_at_maximum_from_2_2_mode(waveform=waveform, memory_generator=memory_generator)
    # time_shift = kwargs.get('time_shift', 0.)
    # time_shift += shift * (series.time_array[1] - series.time_array[0])
    # time_shift = 0
    waveform_fd = nfft(waveform, series.sampling_frequency)
    for mode in waveform:
        indexes = np.where(series.frequency_array < kwargs.get('minimum_frequency', 20))
        waveform_fd[mode][indexes] = 0
    # waveform_fd = apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
    #                                                 duration=series.duration, shift=time_shift)
    return waveform_fd


def convert_to_frequency_domain_fast(series, waveform, **kwargs):
    waveform = apply_window(waveform=waveform, times=series.time_array, kwargs=kwargs)
    # _, shift = wrap_at_maximum(waveform=waveform)
    # time_shift = kwargs.get('time_shift', 0.)
    # time_shift += shift * (series.time_array[1] - series.time_array[0])
    # time_shift = 0
    waveform_fd = nfft(waveform, series.sampling_frequency)
    for mode in waveform:
        indexes = np.where(series.frequency_array < kwargs.get('minimum_frequency', 20))
        waveform_fd[mode][indexes] = 0
    # waveform_fd = apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
    #                                                 duration=series.duration, shift=time_shift)
    return waveform_fd

# def convert_to_frequency_domain_phenomxphm(series, waveform, **kwargs):
#     waveform_fd = nfft(waveform, series.sampling_frequency)
#     for mode in waveform:
#         indexes = np.where(series.frequency_array < kwargs.get('minimum_frequency', 20))
#         waveform_fd[mode][indexes] = 0
#     return waveform_fd
