import copy

import bilby
import gwmemory
import numpy as np
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.source import lal_binary_black_hole
from scipy.signal.windows import tukey

from memestr.core.waveforms import apply_window, wrap_at_maximum_memory_generator, wrap_by_n_indices, nfft_vectorizable, \
    apply_time_shift_frequency_domain, wrap_at_maximum, get_alpha, gamma_lmlm
from memestr.core.waveforms.surrogate import _evaluate_hybrid_surrogate


def time_domain_nr_hyb_sur_waveform_with_memory(times, mass_ratio, total_mass, s13, s23,
                                                luminosity_distance, inc, phase, **kwargs):
    waveform, memory, _ = _evaluate_hybrid_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                     luminosity_distance=luminosity_distance, phase=phase,
                                                     s13=s13, s23=s23, kwargs=kwargs)
    for mode in memory:
        waveform[mode] += memory[mode]
    return apply_window(waveform=waveform, times=times, kwargs=kwargs)


def time_domain_nr_hyb_sur_waveform_with_memory_wrapped(times, mass_ratio, total_mass, s13, s23,
                                                        luminosity_distance, inc, phase, **kwargs):
    waveform, memory, memory_generator = _evaluate_hybrid_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                                    luminosity_distance=luminosity_distance, phase=phase,
                                                                    s13=s13, s23=s23, kwargs=kwargs)
    windowed_waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    shift = kwargs.get('shift', None)
    if not shift:
        # Do windowing and shifting separately without memory in order to be consistent with waveform without memory case
        _, shift = wrap_at_maximum_memory_generator(waveform=windowed_waveform, memory_generator=memory_generator)

    for mode in memory:
        waveform[mode] += memory[mode]

    waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    return wrap_by_n_indices(shift=int(shift), waveform=waveform)


def frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped(frequencies, mass_ratio, total_mass, s13, s23,
                                                             luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    waveform, memory, memory_generator = _evaluate_hybrid_surrogate(times=series.time_array, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                                    luminosity_distance=luminosity_distance, phase=phase,
                                                                    s13=s13, s23=s23, kwargs=kwargs)
    shift = kwargs.get('shift', None)
    if not shift:
        # Do windowing and shifting separately without memory in order to be consistent with waveform without memory case
        _, shift = wrap_at_maximum_memory_generator(waveform=waveform, memory_generator=memory_generator)

    for mode in memory:
        waveform[mode] += memory[mode]

    waveform = apply_window(waveform=waveform, times=series.time_array, kwargs=kwargs)
    time_shift = shift * (series.time_array[1] - series.time_array[0])

    waveform_fd = nfft_vectorizable(waveform, series.sampling_frequency)
    for mode in ['plus', 'cross']:
        waveform_fd[mode], frequency_array = bilby.core.utils.nfft(waveform[mode], series.sampling_frequency)
        indexes = np.where(frequency_array < 20)
        waveform_fd[mode][indexes] = 0
    return apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=frequencies,
                                             duration=series.duration, shift=time_shift)


def time_domain_nr_hyb_sur_waveform_without_memory_wrapped(times, mass_ratio, total_mass, s13, s23,
                                                           luminosity_distance, inc, phase, **kwargs):
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                            luminosity_distance=luminosity_distance, phase=phase, s13=s13, s23=s23,
                                                            kwargs=kwargs, fold_in_memory=False)
    waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    waveform, shift = wrap_at_maximum_memory_generator(waveform=waveform, memory_generator=memory_generator)
    return waveform, shift


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