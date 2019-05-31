import bilby
import gwmemory
import numpy as np

from memestr.core.waveforms import apply_window, wrap_at_maximum_memory_generator, nfft_vectorizable, \
    apply_time_shift_frequency_domain, wrap_by_n_indices, gamma_lmlm


def frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped(frequencies, mass_ratio, total_mass, s13, s23,
                                                                luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=series.time_array, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                                             luminosity_distance=luminosity_distance, phase=phase, s13=s13, s23=s23,
                                                             kwargs=kwargs, fold_in_memory=False)
    waveform = apply_window(waveform=waveform, times=series.time_array, kwargs=kwargs)
    _, shift = wrap_at_maximum_memory_generator(waveform=waveform, memory_generator=memory_generator)
    time_shift = shift * (series.time_array[1] - series.time_array[0])
    waveform_fd = nfft_vectorizable(waveform, series.sampling_frequency)
    waveform_fd = apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
                                                    duration=series.duration, shift=time_shift)
    for mode in ['plus', 'cross']:
        waveform_fd[mode], frequency_array = bilby.core.utils.nfft(waveform[mode], series.sampling_frequency)
        indexes = np.where(frequency_array < 20)
        waveform_fd[mode][indexes] = 0
    return waveform_fd, shift


def time_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return(times, mass_ratio, total_mass, s13, s23,
                                                                           luminosity_distance, inc, phase, **kwargs):
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=times, mass_ratio=mass_ratio, total_mass=total_mass, s13=s13, s23=s23,
                                                            luminosity_distance=luminosity_distance, inc=inc, phase=phase,
                                                            fold_in_memory=False, kwargs=kwargs)
    waveform = apply_window(waveform, times, kwargs)
    shift = kwargs.get('shift', None)
    if shift:
        return wrap_by_n_indices(shift=int(shift), waveform=waveform)
    else:
        waveform, _ = wrap_at_maximum_memory_generator(waveform, memory_generator)
        return waveform


def frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return(frequencies, mass_ratio, total_mass, s13, s23,
                                                                                luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=series.time_array, mass_ratio=mass_ratio, total_mass=total_mass, s13=s13, s23=s23,
                                                            luminosity_distance=luminosity_distance, inc=inc, phase=phase,
                                                            fold_in_memory=False, kwargs=kwargs)
    waveform = apply_window(waveform, series.time_array, kwargs)
    waveform_fd = nfft_vectorizable(waveform, series.sampling_frequency)
    shift = kwargs.get('shift', None)
    if shift is None:
        _, shift = wrap_at_maximum_memory_generator(waveform=waveform, memory_generator=memory_generator)
    time_shift = shift * (series.time_array[1] - series.time_array[0])
    for mode in ['plus', 'cross']:
        waveform_fd[mode], frequency_array = bilby.core.utils.nfft(waveform[mode], series.sampling_frequency)
        indexes = np.where(frequency_array < 20)
        waveform_fd[mode][indexes] = 0
    return apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
                                             duration=series.duration, shift=time_shift)


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