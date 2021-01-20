import bilby
import gwmemory

from .utils import apply_window, gamma_lmlm, convert_to_frequency_domain


def fd_nr_sur_7dq4(frequencies, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                   luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_surrogate(times=series.time_array, total_mass=total_mass,
                                                     mass_ratio=mass_ratio, inc=inc,
                                                     luminosity_distance=luminosity_distance, phase=phase,
                                                     s11=s11, s12=s12, s13=s13, s21=s21, s22=s22, s23=s23,
                                                     kwargs=kwargs, fold_in_memory=False)
    return convert_to_frequency_domain(memory_generator, series, waveform, inc, phase, **kwargs)


def fd_nr_sur_7dq4_with_memory(frequencies, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                               luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    waveform, memory, memory_generator = _evaluate_surrogate(times=series.time_array, total_mass=total_mass,
                                                             mass_ratio=mass_ratio, inc=inc,
                                                             luminosity_distance=luminosity_distance,
                                                             phase=phase,
                                                             s11=s11, s12=s12, s13=s13, s21=s21, s22=s22, s23=s23,
                                                             kwargs=kwargs)
    for mode in memory:
        waveform[mode] += memory[mode]

    return convert_to_frequency_domain(memory_generator, series, waveform, inc, phase, **kwargs)


def fd_nr_sur_7dq4_memory_only(frequencies, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                               luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    _, memory, memory_generator = _evaluate_surrogate(times=series.time_array, total_mass=total_mass,
                                                      mass_ratio=mass_ratio, inc=inc,
                                                      luminosity_distance=luminosity_distance,
                                                      phase=phase, s11=s11, s12=s12, s13=s13, s21=s21, s22=s22, s23=s23,
                                                      kwargs=kwargs)

    return convert_to_frequency_domain(memory_generator, series, memory, inc, phase, **kwargs)


def td_nr_sur_7dq4_memory_only(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                               luminosity_distance, inc, phase, **kwargs):
    _, memory, _ = _evaluate_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                       luminosity_distance=luminosity_distance, phase=phase,
                                       s11=s11, s12=s12, s13=s13, s21=s21, s22=s22, s23=s23, kwargs=kwargs,
                                       fold_in_memory=True)
    return apply_window(waveform=memory, times=times, kwargs=kwargs)


def td_nr_sur_7dq4(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                   luminosity_distance, inc, phase, **kwargs):
    waveform, memory = _evaluate_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio,
                                           inc=inc, luminosity_distance=luminosity_distance, phase=phase,
                                           s11=s11, s12=s12, s13=s13, s21=s21, s22=s22, s23=s23,
                                           kwargs=kwargs, fold_in_memory=False)
    return waveform


def td_nr_sur_7dq4_with_memory(times, mass_ratio, total_mass, s11, s12, s13, s21, s22, s23,
                               luminosity_distance, inc, phase, **kwargs):
    waveform, memory, _ = _evaluate_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio,
                                              inc=inc, luminosity_distance=luminosity_distance, phase=phase,
                                              s11=s11, s12=s12, s13=s13, s21=s21, s22=s22, s23=s23,
                                              kwargs=kwargs, fold_in_memory=True)
    for mode in waveform:
        waveform[mode] += memory[mode]
    return waveform


def _evaluate_surrogate(times, total_mass, mass_ratio, inc, luminosity_distance, phase, s11, s12, s13, s21, s22, s23,
                        kwargs, fold_in_memory=True):
    memory_generator = gwmemory.waveforms.NRSur7dq4(q=mass_ratio,
                                                    total_mass=total_mass,
                                                    spin_1x=s11, spin_1y=s12, spin_1z=s13,
                                                    spin_2x=s21, spin_2y=s22, spin_2z=s23,
                                                    times=times,
                                                    distance=luminosity_distance,
                                                    minimum_frequency=kwargs.get('minimum_frequency', 10),
                                                    units='mks'
                                                    )

    oscillatory, _ = memory_generator.time_domain_oscillatory(times=times, inc=inc, phase=phase)
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator
