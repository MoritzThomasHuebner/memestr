import copy
from copy import deepcopy

import bilby
import numpy as np

from .. import gwmemory
from ..utils import convert_to_frequency_domain, convert_to_frequency_domain_with_memory, apply_window, gamma_lmlm


def fd_imrx_with_memory(frequencies, mass_ratio, total_mass, luminosity_distance,
                        s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory, memory_generator = _evaluate_imrx(series.time_array, total_mass=total_mass,
                                                        mass_ratio=mass_ratio, inc=inc,
                                                        luminosity_distance=luminosity_distance, phase=phase,
                                                        s13=s13, s23=s23, fold_in_memory=True)
    reference_waveform = deepcopy(waveform)
    return convert_to_frequency_domain_with_memory(series, waveform, reference_waveform, **kwargs)


def fd_imrx(frequency_array, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequency_array
    waveform, memory_generator = _evaluate_imrx(series.time_array, total_mass=total_mass,
                                                mass_ratio=mass_ratio, inc=inc,
                                                luminosity_distance=luminosity_distance, phase=phase,
                                                s13=s13, s23=s23, fold_in_memory=False)
    return convert_to_frequency_domain(series, waveform, **kwargs)


def fd_imrx_fast(frequency_array, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequency_array
    waveform = _evaluate_imrx_fast(series.time_array, total_mass=total_mass,
                                   mass_ratio=mass_ratio, inc=inc,
                                   luminosity_distance=luminosity_distance, phase=phase,
                                   s13=s13, s23=s23)
    return convert_to_frequency_domain(series, waveform, **kwargs)


def fd_imrx_select_modes(frequency_array, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequency_array
    modes = kwargs.get('modes')
    waveform, memory_generator = _evaluate_imrx(series.time_array, total_mass=total_mass,
                                                mass_ratio=mass_ratio, inc=inc,
                                                luminosity_distance=luminosity_distance, phase=phase,
                                                s13=s13, s23=s23, fold_in_memory=False, modes=modes)
    return convert_to_frequency_domain(series, waveform, **kwargs)


def fd_imrx_22(frequency_array, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase, **kwargs):
    return fd_imrx_select_modes(frequency_array, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase,
                                modes=[(2, 2), (2, -2)], **kwargs)


def fd_imrx_22_with_memory(frequencies, mass_ratio, total_mass, luminosity_distance,
                           s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory, memory_generator = _evaluate_imrx(series.time_array, total_mass=total_mass,
                                                        mass_ratio=mass_ratio, inc=inc,
                                                        luminosity_distance=luminosity_distance, phase=phase,
                                                        s13=s13, s23=s23, fold_in_memory=True, modes=[(2, 2), (2, -2)])
    reference_waveform = deepcopy(waveform)
    for mode in memory:
        waveform[mode] += memory[mode]
    return convert_to_frequency_domain_with_memory(series, waveform, reference_waveform, **kwargs)


def fd_imrx_memory_only(frequencies, mass_ratio, total_mass, luminosity_distance,
                        s13, s23, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory, memory_generator = _evaluate_imrx(series.time_array, total_mass=total_mass,
                                                        mass_ratio=mass_ratio, inc=inc,
                                                        luminosity_distance=luminosity_distance, phase=phase,
                                                        s13=s13, s23=s23, fold_in_memory=True)
    return convert_to_frequency_domain_with_memory(series, memory, waveform, **kwargs)


def td_imrx_with_memory(times, mass_ratio, total_mass, luminosity_distance, s13,
                        s23, inc, phase, **kwargs):
    waveform, memory, _ = _evaluate_imrx(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                         luminosity_distance=luminosity_distance, phase=phase,
                                         s13=s13, s23=s23, fold_in_memory=True)
    for mode in waveform:
        waveform[mode] += memory[mode]
    return waveform


def td_imrx(times, mass_ratio, total_mass, luminosity_distance,
            s13, s23, inc, phase, **kwargs):
    waveform, _ = _evaluate_imrx(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                 luminosity_distance=luminosity_distance, phase=phase,
                                 s13=s13, s23=s23, fold_in_memory=False)
    return waveform


def td_imrx_fast(times, mass_ratio, total_mass, luminosity_distance,
                 s13, s23, inc, phase, **kwargs):
    hpc = _evaluate_imrx_fast(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                              luminosity_distance=luminosity_distance, phase=phase,
                              s13=s13, s23=s23)
    return hpc

def td_imrx_memory_only(times, mass_ratio, total_mass, luminosity_distance,
                        s13, s23, inc, phase, **kwargs):
    _, memory, _ = _evaluate_imrx(times=times, total_mass=total_mass,
                                  mass_ratio=mass_ratio, inc=inc,
                                  luminosity_distance=luminosity_distance, phase=phase,
                                  s13=s13, s23=s23, fold_in_memory=True)
    return apply_window(waveform=memory, times=times, kwargs=kwargs)


def td_imrx_22_with_memory(times, mass_ratio, total_mass, luminosity_distance, s13,
                           s23, inc, phase, **kwargs):
    waveform, memory, _ = _evaluate_imrx(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                         luminosity_distance=luminosity_distance, phase=phase,
                                         s13=s13, s23=s23, fold_in_memory=True, modes=[(2, 2), (2, -2)])
    for mode in waveform:
        waveform[mode] += memory[mode]
    return waveform


def td_imrx_22(times, mass_ratio, total_mass, luminosity_distance,
               s13, s23, inc, phase, **kwargs):
    waveform, _ = _evaluate_imrx(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                 luminosity_distance=luminosity_distance, phase=phase,
                                 s13=s13, s23=s23, fold_in_memory=False, modes=[(2, 2), (2, -2)])
    return waveform


def _evaluate_imrx(times, total_mass, mass_ratio, inc, luminosity_distance, phase,
                   s13, s23, fold_in_memory=True, modes=None):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.PhenomXHM(q=mass_ratio,
                                                    m_tot=total_mass,
                                                    distance=luminosity_distance,
                                                    s1=np.array([0., 0., s13]),
                                                    s2=np.array([0., 0., s23]),
                                                    times=temp_times)
    oscillatory = memory_generator.time_domain_oscillatory(inc=inc, phase=phase, modes=modes)
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator


def _evaluate_imrx_fast(times, total_mass, mass_ratio, inc, luminosity_distance, phase,
                        s13, s23):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.PhenomXHM(q=mass_ratio,
                                                    m_tot=total_mass,
                                                    distance=luminosity_distance,
                                                    s1=np.array([0., 0., s13]),
                                                    s2=np.array([0., 0., s23]),
                                                    times=temp_times)
    return memory_generator.time_domain_oscillatory_from_polarisations(inc=inc, phase=phase)

