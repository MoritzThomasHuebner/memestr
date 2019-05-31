import copy

import gwmemory
import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from scipy.signal.windows import tukey
from .utils import apply_window, wrap_at_maximum, get_alpha, gamma_lmlm


def frequency_domain_IMRPhenomD_waveform_without_memory(frequencies, mass_ratio, total_mass, luminosity_distance,
                                                        s13, s23, inc, phase,
                                                        **kwargs):
    parameters = dict(mass_ratio=mass_ratio, total_mass=total_mass, luminosity_distance=luminosity_distance,
                      theta_jn=inc, phase=phase, chi_1=s13, chi_2=s23)
    parameters, _ = convert_to_lal_binary_black_hole_parameters(parameters)
    return lal_binary_black_hole(frequency_array=frequencies, **parameters, **kwargs)


def time_domain_IMRPhenomD_waveform_with_memory(times, mass_ratio, total_mass, luminosity_distance, s11, s12, s13,
                                                s21, s22, s23, inc, phase, **kwargs):
    waveform = _evaluate_imr_phenom_d_with_memory(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                  luminosity_distance=luminosity_distance, phase=phase, s11=s11,
                                                  s12=s12, s13=s13, s21=s21, s22=s22, s23=s23)

    return apply_window(waveform=waveform, times=times, kwargs=kwargs)


def time_domain_IMRPhenomD_waveform_with_memory_wrapped(times, mass_ratio, total_mass, luminosity_distance, s11, s12,
                                                        s13,
                                                        s21, s22, s23, inc, phase, **kwargs):
    waveform = _evaluate_imr_phenom_d_with_memory(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                  luminosity_distance=luminosity_distance, phase=phase, s11=s11,
                                                  s12=s12, s13=s13, s21=s21, s22=s22, s23=s23)
    waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    return wrap_at_maximum(waveform=waveform)


def time_domain_IMRPhenomD_waveform_without_memory(times, mass_ratio, total_mass, luminosity_distance, s11, s12, s13,
                                                   s21, s22, s23, inc, phase, **kwargs):
    waveform = _evaluate_imr_phenom_d_without_memory(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                     luminosity_distance=luminosity_distance, phase=phase, s11=s11,
                                                     s12=s12, s13=s13, s21=s21, s22=s22, s23=s23)
    return apply_window(waveform=waveform, times=times, kwargs=kwargs)


def time_domain_IMRPhenomD_waveform_without_memory_wrapped(times, mass_ratio, total_mass, luminosity_distance,
                                                           s11, s12, s13, s21, s22, s23, inc, phase, **kwargs):
    waveform = _evaluate_imr_phenom_d_without_memory(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                     luminosity_distance=luminosity_distance, phase=phase, s11=s11,
                                                     s12=s12, s13=s13, s21=s21, s22=s22, s23=s23)
    waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    return wrap_at_maximum(waveform=waveform)


def time_domain_IMRPhenomD_memory_waveform(times, mass_ratio, total_mass, luminosity_distance, s11, s12, s13,
                                           s21, s22, s23, inc, phase, **kwargs):
    temp_times = copy.copy(times)
    wave = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                          q=mass_ratio,
                                          MTot=total_mass,
                                          distance=luminosity_distance,
                                          S1=np.array([s11, s12, s13]),
                                          S2=np.array([s21, s22, s23]),
                                          times=temp_times)
    alpha = get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)
    memory, _ = wave.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    for mode in memory:
        memory[mode] *= window
    return memory


def _evaluate_imr_phenom_d_without_memory(times, total_mass, mass_ratio, inc, luminosity_distance, phase, s11, s12, s13,
                                          s21, s22, s23):
    temp_times = copy.copy(times)
    wave = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                          q=mass_ratio,
                                          MTot=total_mass,
                                          distance=luminosity_distance,
                                          S1=np.array([s11, s12, s13]),
                                          S2=np.array([s21, s22, s23]),
                                          times=temp_times)
    waveform = wave.time_domain_oscillatory(inc=inc, phase=phase)
    return waveform


def _evaluate_imr_phenom_d_with_memory(times, total_mass, mass_ratio, inc, luminosity_distance, phase, s11, s12, s13,
                                       s21, s22, s23):
    temp_times = copy.copy(times)
    wave = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                          q=mass_ratio,
                                          MTot=total_mass,
                                          distance=luminosity_distance,
                                          S1=np.array([s11, s12, s13]),
                                          S2=np.array([s21, s22, s23]),
                                          times=temp_times)
    oscillatory = wave.time_domain_oscillatory(inc=inc, phase=phase)
    memory, _ = wave.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    waveform = dict()
    for mode in memory:
        waveform[mode] = (memory[mode] + oscillatory[mode])
    return waveform