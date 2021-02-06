import copy
from copy import deepcopy

import bilby
import gwmemory
import numpy as np

from ..utils import convert_to_frequency_domain, apply_window, gamma_lmlm, convert_to_frequency_domain_with_memory


def fd_imrd_with_memory(frequencies, mass_ratio, total_mass, luminosity_distance,
                        s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory, memory_generator = _evaluate_imrd(series.time_array, total_mass=total_mass,
                                                        mass_ratio=mass_ratio, inc=inc,
                                                        luminosity_distance=luminosity_distance, phase=phase,
                                                        s13=s13, s23=s23, fold_in_memory=True)
    reference_waveform = deepcopy(waveform)
    for mode in memory:
        waveform[mode] += memory[mode]
    return convert_to_frequency_domain_with_memory(series, waveform, reference_waveform, **kwargs)


def fd_imrd_memory_only(frequencies, mass_ratio, total_mass, luminosity_distance,
                        s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory, memory_generator = _evaluate_imrd(series.time_array, total_mass=total_mass,
                                                        mass_ratio=mass_ratio, inc=inc,
                                                        luminosity_distance=luminosity_distance, phase=phase,
                                                        s13=s13, s23=s23, fold_in_memory=True)
    return convert_to_frequency_domain_with_memory(series, memory, waveform, **kwargs)


def fd_imrd(frequencies, mass_ratio, total_mass, luminosity_distance,
            s13, s23,
            inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_imrd(series.time_array, total_mass=total_mass,
                                                mass_ratio=mass_ratio, inc=inc,
                                                luminosity_distance=luminosity_distance, phase=phase,
                                                s13=s13, s23=s23, fold_in_memory=False)
    return convert_to_frequency_domain(series, waveform, **kwargs)


def td_imrd_with_memory(times, mass_ratio, total_mass, luminosity_distance, s13,
                        s23, inc, phase, **kwargs):
    waveform, memory, _ = _evaluate_imrd(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                         luminosity_distance=luminosity_distance, phase=phase,
                                         s13=s13, s23=s23, fold_in_memory=True)
    for mode in waveform:
        waveform[mode] += memory[mode]
    return apply_window(waveform=waveform, times=times, kwargs=kwargs)


def td_imrd(times, mass_ratio, total_mass, luminosity_distance,
            s13, s23, inc, phase, **kwargs):
    waveform, _ = _evaluate_imrd(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                 luminosity_distance=luminosity_distance, phase=phase,
                                 s13=s13, s23=s23, fold_in_memory=False)
    return apply_window(waveform=waveform, times=times, kwargs=kwargs)


def td_imrd_memory_only(times, mass_ratio, total_mass, luminosity_distance,
                        s13, s23, inc, phase, **kwargs):
    _, memory, _ = _evaluate_imrd(times=times, total_mass=total_mass,
                                  mass_ratio=mass_ratio, inc=inc,
                                  luminosity_distance=luminosity_distance, phase=phase,
                                  s13=s13, s23=s23, fold_in_memory=True)
    return apply_window(waveform=memory, times=times, kwargs=kwargs)


def _evaluate_imrd(times, total_mass, mass_ratio, inc, luminosity_distance, phase,
                   s13, s23, fold_in_memory=True):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                                      q=mass_ratio,
                                                      MTot=total_mass,
                                                      distance=luminosity_distance,
                                                      S1=np.array([0., 0., s13]),
                                                      S2=np.array([0., 0., s23]),
                                                      times=temp_times)
    oscillatory = memory_generator.time_domain_oscillatory(inc=inc, phase=phase)
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator