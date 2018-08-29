import numpy as np
import gwmemory
import copy

gamma_lmlm = gwmemory.angles.load_gamma()


def time_domain_nr_sur_waveform_without_memory(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                                               luminosity_distance,
                                               inc, phase, LMax, **kwargs):
    temp_times = copy.copy(times)
    print(s11)
    print(s12)
    print(s13)
    print(s21)
    print(s22)
    print(s23)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    LMax=LMax,
                                                    times=temp_times,
                                                    distance=luminosity_distance
                                                    )
    h_oscillatory, _ = memory_generator.time_domain_oscillatory(times=temp_times, inc=inc, phase=phase)
    return h_oscillatory


def time_domain_nr_sur_waveform_with_memory(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                                            luminosity_distance,
                                            inc, phase, LMax, **kwargs):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    LMax=LMax,
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
                                       inc, phase, LMax, **kwargs):
    temp_times = copy.copy(times)
    memory_generator = gwmemory.waveforms.Surrogate(q=mass_ratio,
                                                    name='',
                                                    MTot=total_mass,
                                                    S1=np.array([s11, s12, s13]),
                                                    S2=np.array([s21, s22, s23]),
                                                    LMax=LMax,
                                                    times=temp_times,
                                                    distance=luminosity_distance
                                                    )
    h_memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    return h_memory


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

    res = dict()
    for mode in memory:
        res[mode] = memory[mode] + oscillatory[mode]
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
    oscillatory = wave.time_domain_oscillatory(inc=inc, phase=phase)
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
    memory, _ = wave.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
    return memory
