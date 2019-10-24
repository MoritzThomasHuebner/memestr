import bilby
import gwmemory

from .utils import apply_window, gamma_lmlm, convert_to_frequency_domain


def frequency_domain_nr_hyb_sur_waveform_without_memory(frequencies, mass_ratio, total_mass, s13, s23,
                                                        luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=series.time_array, total_mass=total_mass,
                                                            mass_ratio=mass_ratio, inc=inc,
                                                            luminosity_distance=luminosity_distance, phase=phase,
                                                            s13=s13, s23=s23, kwargs=kwargs, fold_in_memory=False)
    return convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)


def frequency_domain_nr_hyb_sur_waveform_with_memory(frequencies, mass_ratio, total_mass, s13, s23,
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

    return convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)


def frequency_domain_nr_hyb_sur_memory_waveform(frequencies, mass_ratio, total_mass, s13, s23,
                                                luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    _, memory, memory_generator = _evaluate_hybrid_surrogate(times=series.time_array, total_mass=total_mass,
                                                             mass_ratio=mass_ratio, inc=inc,
                                                             luminosity_distance=luminosity_distance,
                                                             phase=phase, s13=s13, s23=s23, kwargs=kwargs)

    return convert_to_frequency_domain(memory_generator, series, memory, **kwargs)


def time_domain_nr_hyb_sur_waveform_memory(times, mass_ratio, total_mass, s13, s23,
                                           luminosity_distance, inc, phase, **kwargs):
    _, memory, _ = _evaluate_hybrid_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                              luminosity_distance=luminosity_distance, phase=phase,
                                              s13=s13, s23=s23, kwargs=kwargs)
    return apply_window(waveform=memory, times=times, kwargs=kwargs)


def time_domain_nr_hyb_sur_waveform_without_memory(times, mass_ratio, total_mass, s13, s23,
                                                   luminosity_distance, inc, phase, **kwargs):
    waveform, memory_generator = _evaluate_hybrid_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio,
                                                            inc=inc,
                                                            luminosity_distance=luminosity_distance, phase=phase,
                                                            s13=s13, s23=s23,
                                                            kwargs=kwargs, fold_in_memory=False)
    return waveform


def _evaluate_hybrid_surrogate(times, total_mass, mass_ratio, inc, luminosity_distance, phase, s13, s23, kwargs,
                               fold_in_memory=True):
    memory_generator = gwmemory.waveforms.HybridSurrogate(q=mass_ratio,
                                                          total_mass=total_mass,
                                                          spin_1=s13,
                                                          spin_2=s23,
                                                          times=times,
                                                          distance=luminosity_distance,
                                                          minimum_frequency=kwargs.get('minimum_frequency', 10),
                                                          sampling_frequency=kwargs.get('sampling_frequency', 2048),
                                                          units='mks'
                                                          )

    oscillatory, _ = memory_generator.time_domain_oscillatory(times=times, inc=inc, phase=phase)
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator
