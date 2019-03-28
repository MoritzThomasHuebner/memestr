import numpy as np
import gwmemory
import copy
from scipy.signal.windows import tukey
from bilby.gw.source import lal_binary_black_hole

gamma_lmlm = gwmemory.angles.load_gamma()
roll_off = 0.2


def time_domain_nr_sur_waveform_without_memory_base_modes(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                                                          luminosity_distance,
                                                          inc, phase, **kwargs):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    modes=[(2, 2), (2, -2)],
                                                    LMax=2,
                                                    times=temp_times,
                                                    distance=luminosity_distance
                                                    )
    h_oscillatory, _ = memory_generator.time_domain_oscillatory(times=temp_times, inc=inc, phase=phase)
    return h_oscillatory


def time_domain_nr_sur_waveform_with_memory_base_modes(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                                                       luminosity_distance,
                                                       inc, phase, **kwargs):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    modes=[(2, 2), (2, -2)],
                                                    LMax=2,
                                                    times=temp_times,
                                                    distance=luminosity_distance
                                                    )
    h_oscillatory, _ = memory_generator.time_domain_oscillatory(times=temp_times, inc=inc, phase=phase)
    h_memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    res = dict()
    for mode in h_memory:
        res[mode] = h_memory[mode] + h_oscillatory[mode]
    return res


def time_domain_nr_sur_waveform_without_memory(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                                               luminosity_distance,
                                               inc, phase, **kwargs):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    LMax=kwargs['l_max'],
                                                    times=temp_times,
                                                    distance=luminosity_distance
                                                    )
    h_oscillatory, _ = memory_generator.time_domain_oscillatory(times=temp_times, inc=inc, phase=phase)
    return h_oscillatory


def time_domain_nr_sur_waveform_with_memory(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                                            luminosity_distance,
                                            inc, phase, **kwargs):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    LMax=kwargs['l_max'],
                                                    times=temp_times,
                                                    distance=luminosity_distance
                                                    )
    h_oscillatory, _ = memory_generator.time_domain_oscillatory(times=temp_times, inc=inc, phase=phase)
    h_memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    res = dict()
    for mode in h_memory:
        res[mode] = h_memory[mode] + h_oscillatory[mode]
    return res


def time_domain_nr_sur_memory_waveform(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23, luminosity_distance,
                                       inc, phase, **kwargs):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    LMax=kwargs['l_max'],
                                                    times=temp_times,
                                                    distance=luminosity_distance
                                                    )
    h_memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    return h_memory


def frequency_domain_IMRPhenomD_waveform_without_memory(frequencies, mass_ratio, total_mass, luminosity_distance,
                                                        s11, s12, s13, s21, s22, s23, inc, phase,
                                                        **kwargs):
    frequencies = copy.copy(frequencies)
    mass_1 = total_mass / (1 + mass_ratio)
    mass_2 = mass_1 * mass_ratio
    return lal_binary_black_hole(frequency_array=frequencies, mass_1=mass_1, mass_2=mass_2,
                                 luminosity_distance=luminosity_distance, theta_jn=inc, phase=phase, a_1=0,
                                 a_2=0, tilt_1=0, tilt_2=0, phi_12=0, phi_jl=0, **kwargs)


def time_domain_IMRPhenomD_waveform_with_memory(times, mass_ratio, total_mass, luminosity_distance, s11, s12, s13,
                                                s21, s22, s23, inc, phase, **kwargs):
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
    alpha = _get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)
    res = dict()
    for mode in memory:
        res[mode] = (memory[mode] + oscillatory[mode]) * window
    return res

def time_domain_IMRPhenomD_waveform_with_memory_open_data(times, mass_1, mass_2, luminosity_distance,
                                                iota, phase, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, **kwargs):
    temp_times = copy.copy(times)
    wave = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                          q=mass_1/mass_2,
                                          MTot=mass_1+mass_2,
                                          distance=luminosity_distance,
                                          S1=np.array([0.0, 0.0, 0.0]),
                                          S2=np.array([0.0, 0.0, 0.0]),
                                          times=temp_times)
    oscillatory = wave.time_domain_oscillatory(inc=iota, phase=phase)
    memory, _ = wave.time_domain_memory(inc=iota, phase=phase, gamma_lmlm=gamma_lmlm)
    alpha = _get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)
    res = dict()
    for mode in memory:
        res[mode] = (memory[mode] + oscillatory[mode]) * window
    return res

def time_domain_IMRPhenomD_waveform_without_memory(times, mass_ratio, total_mass, luminosity_distance, s11, s12, s13,
                                                   s21, s22, s23, inc, phase, **kwargs):
    temp_times = copy.copy(times)
    wave = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                          q=mass_ratio,
                                          MTot=total_mass,
                                          distance=luminosity_distance,
                                          S1=np.array([s11, s12, s13]),
                                          S2=np.array([s21, s22, s23]),
                                          times=temp_times)
    alpha = _get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)
    oscillatory = wave.time_domain_oscillatory(inc=inc, phase=phase)
    for mode in oscillatory:
        oscillatory[mode] *= window
    return oscillatory


def time_domain_IMRPhenomD_waveform_without_memory_wrapped(times, mass_ratio, total_mass, luminosity_distance,
                                                           s11, s12, s13, s21, s22, s23, inc, phase, **kwargs):
    temp_times = copy.copy(times)
    wave = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                          q=mass_ratio,
                                          MTot=total_mass,
                                          distance=luminosity_distance,
                                          S1=np.array([s11, s12, s13]),
                                          S2=np.array([s21, s22, s23]),
                                          times=temp_times)
    alpha = _get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)
    oscillatory = wave.time_domain_oscillatory(inc=inc, phase=phase)
    for mode in oscillatory:
        oscillatory[mode] *= window
    max_index = np.argmax(np.abs(np.abs(oscillatory['plus']) + np.abs(oscillatory['cross'])))
    max_index_fd_model = kwargs.get('max_index_fd_model', len(oscillatory['plus']))
    oscillatory['plus'] = np.roll(oscillatory['plus'], shift=max_index_fd_model - max_index)
    oscillatory['cross'] = np.roll(oscillatory['cross'], shift=max_index_fd_model - max_index)
    return oscillatory


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
    alpha = _get_alpha(kwargs, times)
    window = tukey(M=len(times), alpha=alpha)
    memory, _ = wave.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    for mode in memory:
        memory[mode] *= window
    return memory


def _get_alpha(kwargs, times):
    if 'alpha' in kwargs:
        if kwargs['alpha']:
            return kwargs['alpha']
    return roll_off / (times[-1] - times[0])
