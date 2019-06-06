import bilby
import gwmemory
import numpy as np

from .utils import apply_window, wrap_at_maximum_from_2_2_mode, nfft, \
    apply_time_shift_frequency_domain, wrap_by_n_indices, gamma_lmlm


def frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped(frequencies, mass_ratio, total_mass, s13, s23,
                                                                luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=series.time_array, total_mass=total_mass,
                                                            mass_ratio=mass_ratio, inc=inc,
                                                            luminosity_distance=luminosity_distance, phase=phase,
                                                            s13=s13, s23=s23, kwargs=kwargs, fold_in_memory=False)
    waveform_fd, shift = convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)
    return waveform_fd, shift


def frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return(frequencies, mass_ratio, total_mass,
                                                                                s13, s23,
                                                                                luminosity_distance, inc, phase,
                                                                                **kwargs):
    waveform, shift = frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped(
        frequencies=frequencies, mass_ratio=mass_ratio, total_mass=total_mass, s13=s13, s23=s23,
        luminosity_distance=luminosity_distance, inc=inc, phase=phase, **kwargs)
    return waveform


def frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped(frequencies, mass_ratio, total_mass, s13, s23,
                                                             luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    waveform, memory, memory_generator = _evaluate_hybrid_surrogate(times=series.time_array, total_mass=total_mass,
                                                                    mass_ratio=mass_ratio, inc=inc,
                                                                    luminosity_distance=luminosity_distance,
                                                                    phase=phase,
                                                                    s13=s13, s23=s23, kwargs=kwargs)
    for mode in memory:
        waveform[mode] += memory[mode]

    waveform_fd, shift = convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)
    return waveform_fd


def convert_to_frequency_domain(memory_generator, series, waveform, **kwargs):
    waveform = apply_window(waveform=waveform, times=series.time_array, kwargs=kwargs)
    _, shift = wrap_at_maximum_from_2_2_mode(waveform=waveform, memory_generator=memory_generator)
    time_shift = kwargs.get('time_shift', 0.)
    time_shift += shift * (series.time_array[1] - series.time_array[0])
    waveform_fd = nfft(waveform, series.sampling_frequency)
    for mode in ['plus', 'cross']:
        indexes = np.where(series.frequency_array < 20)
        waveform_fd[mode][indexes] = 0
    waveform_fd = apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
                                                    duration=series.duration, shift=time_shift)
    return waveform_fd, shift


def time_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return(times, mass_ratio, total_mass, s13, s23,
                                                                           luminosity_distance, inc, phase, **kwargs):
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=times, mass_ratio=mass_ratio, total_mass=total_mass,
                                                            s13=s13, s23=s23,
                                                            luminosity_distance=luminosity_distance, inc=inc,
                                                            phase=phase,
                                                            fold_in_memory=False, kwargs=kwargs)
    waveform = apply_window(waveform, times, kwargs)
    shift = kwargs.get('shift', None)
    if shift:
        return wrap_by_n_indices(shift=int(shift), waveform=waveform)
    else:
        waveform, _ = wrap_at_maximum_from_2_2_mode(waveform, memory_generator)
        return waveform


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
    waveform, memory, memory_generator = _evaluate_hybrid_surrogate(times=times, total_mass=total_mass,
                                                                    mass_ratio=mass_ratio, inc=inc,
                                                                    luminosity_distance=luminosity_distance,
                                                                    phase=phase,
                                                                    s13=s13, s23=s23, kwargs=kwargs)
    windowed_waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    shift = kwargs.get('shift', None)
    if not shift:
        # Do windowing and shifting separately without memory in order to be consistent with waveform without memory case
        _, shift = wrap_at_maximum_from_2_2_mode(waveform=windowed_waveform, memory_generator=memory_generator)

    for mode in memory:
        waveform[mode] += memory[mode]

    waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    return wrap_by_n_indices(shift=int(shift), waveform=waveform)


def time_domain_nr_hyb_sur_waveform_without_memory_wrapped(times, mass_ratio, total_mass, s13, s23,
                                                           luminosity_distance, inc, phase, **kwargs):
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio,
                                                            inc=inc,
                                                            luminosity_distance=luminosity_distance, phase=phase,
                                                            s13=s13, s23=s23,
                                                            kwargs=kwargs, fold_in_memory=False)
    waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    waveform, shift = wrap_at_maximum_from_2_2_mode(waveform=waveform, memory_generator=memory_generator)
    return waveform, shift


def _evaluate_hybrid_surrogate(times, total_mass, mass_ratio, inc, luminosity_distance, phase, s13, s23, kwargs,
                               fold_in_memory=True):
    memory_generator = gwmemory.waveforms.HybridSurrogate(q=mass_ratio,
                                                          total_mass=total_mass,
                                                          spin_1=s13,
                                                          spin_2=s23,
                                                          times=times,
                                                          distance=luminosity_distance,
                                                          minimum_frequency=kwargs.get('mininum_frequency', 10),
                                                          sampling_frequency=kwargs.get('sampling_frequency', 2048),
                                                          units='mks'
                                                          )

    oscillatory, _ = memory_generator.time_domain_oscillatory(times=times, inc=inc, phase=phase)
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator
