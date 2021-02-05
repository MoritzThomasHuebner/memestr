import bilby
import gwmemory

from .utils import apply_window, gamma_lmlm, convert_to_frequency_domain


def fd_nr_sur_7dq4(frequencies, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                   luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=mass_ratio, total_mass=total_mass)
    params = dict(a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
                  phi_jl=phi_jl, theta_jn=inc, mass_1=mass_1, mass_2=mass_2, mass_ratio=mass_ratio,
                  reference_frequency=kwargs.get('reference_frequency', 50), phase=phase)
    params = bilby.gw.conversion.generate_spin_parameters(params)

    waveform, memory_generator = _evaluate_surrogate(times=series.time_array, total_mass=total_mass,
                                                     mass_ratio=mass_ratio, inc=inc,
                                                     luminosity_distance=luminosity_distance, phase=phase,
                                                     s11=params['spin_1x'], s12=params['spin_1y'],
                                                     s13=params['spin_1z'], s21=params['spin_2x'],
                                                     s22=params['spin_2y'], s23=params['spin_2z'],
                                                     kwargs=kwargs, fold_in_memory=False)
    return convert_to_frequency_domain(series, waveform, **kwargs)


def fd_nr_sur_7dq4_with_memory(frequencies, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                               luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=mass_ratio, total_mass=total_mass)
    params = dict(a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
                  phi_jl=phi_jl, theta_jn=inc, mass_1=mass_1, mass_2=mass_2, mass_ratio=mass_ratio,
                  reference_frequency=kwargs.get('reference_frequency', 50), phase=phase)
    params = bilby.gw.conversion.generate_spin_parameters(params)

    waveform, memory, memory_generator = _evaluate_surrogate(times=series.time_array, total_mass=total_mass,
                                                             mass_ratio=mass_ratio, inc=inc,
                                                             luminosity_distance=luminosity_distance,
                                                             phase=phase,
                                                             s11=params['spin_1x'], s12=params['spin_1y'],
                                                             s13=params['spin_1z'], s21=params['spin_2x'],
                                                             s22=params['spin_2y'], s23=params['spin_2z'],
                                                             kwargs=kwargs)
    for mode in memory:
        waveform[mode] += memory[mode]

    return convert_to_frequency_domain(series, waveform, **kwargs)


def fd_nr_sur_7dq4_memory_only(frequencies, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                               luminosity_distance, inc, phase, **kwargs):
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
    series.frequency_array = frequencies

    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=mass_ratio, total_mass=total_mass)
    params = dict(a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
                  phi_jl=phi_jl, theta_jn=inc, mass_1=mass_1, mass_2=mass_2, mass_ratio=mass_ratio,
                  reference_frequency=kwargs.get('reference_frequency', 50), phase=phase)
    params = bilby.gw.conversion.generate_spin_parameters(params)

    _, memory, memory_generator = _evaluate_surrogate(times=series.time_array, total_mass=total_mass,
                                                      mass_ratio=mass_ratio, inc=inc,
                                                      luminosity_distance=luminosity_distance,
                                                      s11=params['spin_1x'], s12=params['spin_1y'],
                                                      s13=params['spin_1z'], s21=params['spin_2x'],
                                                      s22=params['spin_2y'], s23=params['spin_2z'],
                                                      kwargs=kwargs)

    return convert_to_frequency_domain(series, memory, **kwargs)


def td_nr_sur_7dq4_memory_only(times, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                               luminosity_distance, inc, phase, **kwargs):
    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=mass_ratio, total_mass=total_mass)
    params = dict(a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
                  phi_jl=phi_jl, theta_jn=inc, mass_1=mass_1, mass_2=mass_2, mass_ratio=mass_ratio,
                  reference_frequency=kwargs.get('reference_frequency', 50), phase=phase)
    params = bilby.gw.conversion.generate_spin_parameters(params)

    _, memory, _ = _evaluate_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio, inc=inc,
                                       luminosity_distance=luminosity_distance, phase=phase,
                                       s11=params['spin_1x'], s12=params['spin_1y'],
                                       s13=params['spin_1z'], s21=params['spin_2x'],
                                       s22=params['spin_2y'], s23=params['spin_2z'], kwargs=kwargs,
                                       fold_in_memory=True)
    return apply_window(waveform=memory, times=times, kwargs=kwargs)


def td_nr_sur_7dq4(times, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                   luminosity_distance, inc, phase, **kwargs):
    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=mass_ratio, total_mass=total_mass)
    params = dict(a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
                  phi_jl=phi_jl, theta_jn=inc, mass_1=mass_1, mass_2=mass_2, mass_ratio=mass_ratio,
                  reference_frequency=kwargs.get('reference_frequency', 50), phase=phase)
    params = bilby.gw.conversion.generate_spin_parameters(params)

    waveform, memory = _evaluate_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio,
                                           inc=inc, luminosity_distance=luminosity_distance, phase=phase,
                                           s11=params['spin_1x'], s12=params['spin_1y'],
                                           s13=params['spin_1z'], s21=params['spin_2x'],
                                           s22=params['spin_2y'], s23=params['spin_2z'],
                                           kwargs=kwargs, fold_in_memory=False)
    return waveform


def td_nr_sur_7dq4_with_memory(times, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl,
                               luminosity_distance, inc, phase, **kwargs):
    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=mass_ratio, total_mass=total_mass)
    params = dict(a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
                  phi_jl=phi_jl, theta_jn=inc, mass_1=mass_1, mass_2=mass_2,
                  reference_frequency=kwargs.get('reference_frequency', 50), phase=phase)
    params = bilby.gw.conversion.generate_spin_parameters(params)

    waveform, memory, _ = _evaluate_surrogate(times=times, total_mass=total_mass, mass_ratio=mass_ratio,
                                              inc=inc, luminosity_distance=luminosity_distance, phase=phase,
                                              s11=params['spin_1x'], s12=params['spin_1y'],
                                              s13=params['spin_1z'], s21=params['spin_2x'],
                                              s22=params['spin_2y'], s23=params['spin_2z'],
                                              kwargs=kwargs, fold_in_memory=True)
    for mode in waveform:
        waveform[mode] += memory[mode]
    return waveform


def _evaluate_surrogate(times, total_mass, mass_ratio, inc, luminosity_distance, phase, s11, s12, s13, s21, s22, s23,
                        kwargs, fold_in_memory=True):
    memory_generator = gwmemory.waveforms.NRSur7dq4(q=mass_ratio,
                                                    total_mass=total_mass,
                                                    S1=[s11, s12, s13],
                                                    S2=[s21, s22, s23],
                                                    times=times,
                                                    distance=luminosity_distance,
                                                    minimum_frequency=kwargs.get('minimum_frequency', 0),
                                                    reference_frequency=kwargs.get('reference_frequency', 50),
                                                    units='mks'
                                                    )

    oscillatory, _ = memory_generator.time_domain_oscillatory(times=times, inc=inc, phase=phase)
    if not fold_in_memory:
        return oscillatory, memory_generator
    else:
        memory, _ = memory_generator.time_domain_memory(inc=inc, phase=phase, gamma_lmlm=gamma_lmlm)
        return oscillatory, memory, memory_generator


def _evaluate_surrogate_fast(times, total_mass, mass_ratio, inc, luminosity_distance, phase,
                             s11, s12, s13, s21, s22, s23, kwargs):
    memory_generator = gwmemory.waveforms.NRSur7dq4(q=mass_ratio,
                                                    total_mass=total_mass,
                                                    S1=[s11, s12, s13],
                                                    S2=[s21, s22, s23],
                                                    times=times,
                                                    distance=luminosity_distance,
                                                    minimum_frequency=kwargs.get('minimum_frequency', 0),
                                                    reference_frequency=kwargs.get('reference_frequency', 50),
                                                    units='mks'
                                                    )

    return memory_generator.time_domain_oscillatory_from_polarisations(inc=inc, phase=phase)
