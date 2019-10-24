import gwmemory
import numpy as np
from scipy.signal.windows import tukey
from bilby.core.series import CoupledTimeAndFrequencySeries

from .utils import nfft


def time_domain_minimum_waveform_model(times, mass_ratio, total_mass, luminosity_distance, inc, phase, **kwargs):
    times -= times[-1]/2
    mwm = gwmemory.waveforms.MWM(q=mass_ratio, MTot=total_mass, distance=luminosity_distance, times=times)
    time_domain_memory, _ = mwm.time_domain_memory(inc=inc, phase=phase)
    alpha = kwargs.get('alpha', 0.1)
    for mode in time_domain_memory:
        time_domain_memory[mode] *= tukey(len(time_domain_memory[mode]), alpha)
    return time_domain_memory


def time_domain_minimum_waveform_model_wrapped(times, mass_ratio, total_mass, luminosity_distance, inc, phase, **kwargs):
    times -= times[-1]/2
    series = CoupledTimeAndFrequencySeries()
    series.time_array = times
    mwm = gwmemory.waveforms.MWM(q=mass_ratio, MTot=total_mass, distance=luminosity_distance, times=times)
    time_domain_memory, _ = mwm.time_domain_memory(inc=inc, phase=phase)
    alpha = kwargs.get('alpha', 0.1)
    for mode in time_domain_memory:
        time_domain_memory[mode] *= tukey(len(time_domain_memory[mode]), alpha)
    ref_index = int(len(times)/2)
    for mode in time_domain_memory:
        time_domain_memory[mode] = np.roll(time_domain_memory[mode], shift=ref_index)
    return time_domain_memory


def frequency_domain_minimum_waveform_model(frequencies, mass_ratio, total_mass, luminosity_distance, inc, phase, **kwargs):
    series = CoupledTimeAndFrequencySeries()
    series.frequency_array = frequencies
    series.start_time = -series.duration/2
    mwm = gwmemory.waveforms.MWM(q=mass_ratio, MTot=total_mass, distance=luminosity_distance, times=series.time_array)
    time_domain_memory, _ = mwm.time_domain_memory(inc=inc, phase=phase)
    alpha = kwargs.get('alpha', 0.1)
    for mode in time_domain_memory:
        time_domain_memory[mode] *= tukey(len(time_domain_memory[mode]), alpha)
    ref_index = int(len(series.time_array)/2)
    for mode in time_domain_memory:
        time_domain_memory[mode] = np.roll(time_domain_memory[mode], shift=ref_index)
    frequency_domain_memory = nfft(time_domain_memory, series.sampling_frequency)

    return frequency_domain_memory

