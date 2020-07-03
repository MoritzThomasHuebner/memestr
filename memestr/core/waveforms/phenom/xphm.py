import copy

import bilby
import gwmemory
import numpy as np

from ..utils import convert_to_frequency_domain, apply_window, gamma_lmlm


def fd_imrxp_with_memory(frequencies, mass_ratio, chirp_mass, luminosity_distance,
                         theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=chirp_mass, mass_ratio=mass_ratio)
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory, memory_generator = _evaluate_imrxp(series.time_array, total_mass=total_mass,
                                                         mass_ratio=mass_ratio, theta_jn=theta_jn,
                                                         luminosity_distance=luminosity_distance, phase=phase,
                                                         phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                                                         phi_12=phi_12, a_1=a_1, a_2=a_2,
                                                         fold_in_memory=True)
    for mode in memory:
        waveform[mode] += memory[mode]
    return convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)


def fd_imrxp_memory_only(frequencies, mass_ratio, chirp_mass, luminosity_distance,
                         theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=chirp_mass, mass_ratio=mass_ratio)
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    _, memory, memory_generator = _evaluate_imrxp(series.time_array, total_mass=total_mass,
                                                  mass_ratio=mass_ratio, theta_jn=theta_jn,
                                                  luminosity_distance=luminosity_distance, phase=phase,
                                                  phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                                                  phi_12=phi_12, a_1=a_1, a_2=a_2,
                                                  fold_in_memory=True)
    return convert_to_frequency_domain(memory_generator, series, memory, **kwargs)


def fd_imrxp(frequencies, mass_ratio, chirp_mass, luminosity_distance,
             theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=chirp_mass, mass_ratio=mass_ratio)
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory_generator = _evaluate_imrxp(series.time_array, total_mass=total_mass,
                                                 mass_ratio=mass_ratio, theta_jn=theta_jn,
                                                 luminosity_distance=luminosity_distance, phase=phase,
                                                 phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                                                 phi_12=phi_12, a_1=a_1, a_2=a_2,
                                                 fold_in_memory=False)
    fd_strain = convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)
    # for mode in fd_strain:
    #     fd_strain[mode] = np.conjugate(fd_strain[mode])
    return fd_strain


def fd_imrxp_select_modes(frequency_array, mass_ratio, chirp_mass, luminosity_distance,
                          theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(chirp_mass=chirp_mass, mass_ratio=mass_ratio)
    modes = kwargs.get('modes')
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequency_array
    waveform, memory_generator = _evaluate_imrxp(
        times=series.time_array, total_mass=total_mass, mass_ratio=mass_ratio, theta_jn=theta_jn,
        luminosity_distance=luminosity_distance, phase=phase,
        phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, fold_in_memory=False, modes=modes)
    return convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)


def fd_imrxp_22(frequency_array, mass_ratio, chirp_mass, luminosity_distance,
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    return fd_imrxp_select_modes(frequency_array, mass_ratio, chirp_mass, luminosity_distance,
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, modes=[(2, 2), (2, -2)], **kwargs)


def fd_imrxp_22_with_memory(frequencies, mass_ratio, chirp_mass, luminosity_distance,
                            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies
    waveform, memory, memory_generator = _evaluate_imrxp(
        times=series.time_array, total_mass=total_mass, mass_ratio=mass_ratio, theta_jn=theta_jn,
        luminosity_distance=luminosity_distance, phase=phase,
        phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, fold_in_memory=True, modes=[(2, 2), (2, -2)])
    for mode in memory:
        waveform[mode] += memory[mode]
    return convert_to_frequency_domain(memory_generator, series, waveform, **kwargs)


def td_imrxp(times, mass_ratio, total_mass, luminosity_distance, theta_jn, phi_jl,
             tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    waveform, _ = _evaluate_imrxp(times=times, total_mass=total_mass, mass_ratio=mass_ratio, theta_jn=theta_jn,
                                  luminosity_distance=luminosity_distance, phase=phase,
                                  phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                                  phi_12=phi_12, a_1=a_1, a_2=a_2, fold_in_memory=False)
    return waveform


def td_imrxp_with_memory(times, mass_ratio, total_mass, luminosity_distance, theta_jn, phi_jl,
                         tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    waveform, memory, _ = _evaluate_imrxp(times=times, total_mass=total_mass, mass_ratio=mass_ratio, theta_jn=theta_jn,
                                          luminosity_distance=luminosity_distance, phase=phase,
                                          phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                                          phi_12=phi_12, a_1=a_1, a_2=a_2,
                                          fold_in_memory=True)
    for mode in waveform:
        waveform[mode] += memory[mode]
    return waveform


def td_imrxp_memory_only(times, mass_ratio, total_mass, luminosity_distance,
                         theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    _, memory, _ = _evaluate_imrxp(times=times, total_mass=total_mass, mass_ratio=mass_ratio, theta_jn=theta_jn,
                                   luminosity_distance=luminosity_distance, phase=phase,
                                   phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                                   phi_12=phi_12, a_1=a_1, a_2=a_2, fold_in_memory=True)
    return apply_window(waveform=memory, times=times, kwargs=kwargs)


def td_imrxp_22_with_memory(times, mass_ratio, total_mass, luminosity_distance,
                            theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    waveform, memory, _ = _evaluate_imrxp(
        times=times, total_mass=total_mass, mass_ratio=mass_ratio, theta_jn=theta_jn,
        luminosity_distance=luminosity_distance, phase=phase,
        phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, fold_in_memory=True, modes=[(2, 2), (2, -2)])
    for mode in waveform:
        waveform[mode] += memory[mode]
    return waveform


def td_imrxp_22(times, mass_ratio, total_mass, luminosity_distance,
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, phase, **kwargs):
    waveform, _ = _evaluate_imrxp(
        times=times, total_mass=total_mass, mass_ratio=mass_ratio, theta_jn=theta_jn,
        luminosity_distance=luminosity_distance, phase=phase,
        phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, fold_in_memory=False, modes=[(2, 2), (2, -2)])
    return waveform


def _evaluate_imrxp(times, total_mass, mass_ratio, luminosity_distance, phase,
                    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
                    fold_in_memory=True, modes=None):
    temp_times = copy.copy(times)
    reference_frequency = 10
    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=mass_ratio, total_mass=total_mass)
    inc, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1, mass_2=mass_2,
        reference_frequency=reference_frequency, phase=phase)
    memory_generator = gwmemory.waveforms.PhenomXPHM(q=mass_ratio,
                                                     MTot=total_mass,
                                                     distance=luminosity_distance,
                                                     S1=np.array([spin_1x, spin_1y, spin_1z]),
                                                     S2=np.array([spin_2x, spin_2y, spin_2z]),
                                                     times=temp_times)
    oscillatory = memory_generator.time_domain_oscillatory(inc=inc, phase=phase, modes=modes)
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator