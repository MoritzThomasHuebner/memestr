import bilby
import gwmemory
from copy import deepcopy

from .utils import apply_window, gamma_lmlm, convert_to_frequency_domain, convert_to_frequency_domain_with_memory


def fd_teob(frequencies, mass_ratio, total_mass, chi_1, chi_2,
            luminosity_distance, inc, phase, ecc, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_teob(
        times=series.time_array, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
        luminosity_distance=luminosity_distance, phase=phase, chi_1=chi_1, chi_2=chi_2,
        kwargs=kwargs, ecc=ecc, fold_in_memory=False)
    return convert_to_frequency_domain(series, waveform, **kwargs)


def fd_teob_with_memory(frequencies, mass_ratio, total_mass, chi_1, chi_2,
                        luminosity_distance, inc, phase, ecc, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    waveform, memory, memory_generator = _evaluate_teob(
        times=series.time_array, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
        luminosity_distance=luminosity_distance, phase=phase, chi_1=chi_1, chi_2=chi_2, ecc=ecc, kwargs=kwargs)
    reference_waveform = deepcopy(waveform)
    for mode in memory:
        waveform[mode] += memory[mode]
    return convert_to_frequency_domain_with_memory(series, waveform, reference_waveform, **kwargs)


def fd_teob_memory_only(frequencies, mass_ratio, total_mass, chi_1, chi_2,
                        luminosity_distance, inc, phase, ecc, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    waveform, memory, memory_generator = _evaluate_teob(
        times=series.time_array, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
        luminosity_distance=luminosity_distance, phase=phase, chi_1=chi_1, chi_2=chi_2, ecc=ecc, kwargs=kwargs)

    return convert_to_frequency_domain_with_memory(series, memory, waveform, **kwargs)


def td_teob_memory_only(times, mass_ratio, total_mass, chi_1, chi_2,
                        luminosity_distance, inc, phase, ecc, **kwargs):

    _, memory, _ = _evaluate_teob(
        times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc, luminosity_distance=luminosity_distance,
        phase=phase, chi_1=chi_1, chi_2=chi_2, kwargs=kwargs, ecc=ecc, fold_in_memory=True)
    return apply_window(waveform=memory, times=times, kwargs=kwargs)


def td_teob(times, mass_ratio, total_mass, chi_1, chi_2,
            luminosity_distance, inc, phase, ecc, **kwargs):
    waveform, _ = _evaluate_teob(
        times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc, luminosity_distance=luminosity_distance,
        phase=phase, chi_1=chi_1, chi_2=chi_2, kwargs=kwargs, ecc=ecc, fold_in_memory=False)
    return waveform


def td_teob_with_memory(times, mass_ratio, total_mass, chi_1, chi_2,
                        luminosity_distance, inc, phase, ecc, **kwargs):
    waveform, memory, _ = _evaluate_teob(
        times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc, luminosity_distance=luminosity_distance,
        phase=phase, chi_1=chi_1, chi_2=chi_2, kwargs=kwargs, ecc=ecc, fold_in_memory=True)
    for mode in waveform:
        waveform[mode] += memory[mode]
    return waveform


def _evaluate_teob(times, total_mass, mass_ratio, inc, luminosity_distance, phase, chi_1, chi_2,
                   kwargs, ecc, fold_in_memory=True):
    memory_generator = gwmemory.waveforms.TEOBResumS(
        times=times, q=mass_ratio, MTot=total_mass, chi_1=chi_1, chi_2=chi_2,
        distance=luminosity_distance, ecc=ecc, minimum_frequency=kwargs.get('minimum_frequency', 10))

    oscillatory = memory_generator.time_domain_oscillatory(inc=inc, phase=phase, modes=kwargs.get('modes', None))
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator
