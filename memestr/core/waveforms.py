import bilby
import numpy as np
import gwmemory
import copy
from scipy.interpolate import CubicSpline
from scipy.signal.windows import tukey
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
gamma_lmlm = gwmemory.angles.load_gamma()
roll_off = 0.2


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


wrap_at_maximum_vectorized = np.vectorize(pyfunc=wrap_at_maximum, excluded=['kwargs'], otypes=[dict])
wrap_by_n_indices_vectorized = np.vectorize(pyfunc=wrap_by_n_indices, excluded=['shift'], otypes=[dict])
apply_window_vectorized = np.vectorize(pyfunc=apply_window, excluded=['times', 'kwargs'], otypes=[dict])
wrap_by_time_shift_continuous_vectorized = np.vectorize(pyfunc=wrap_by_time_shift_continuous,
                                                        excluded=['times', 'time_shift'],
                                                        otypes=[np.ndarray])
_get_new_times_vectorized = np.vectorize(pyfunc=_get_new_times, excluded=['times'])


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