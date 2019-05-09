from copy import deepcopy

import bilby
import bilby.gw.utils as utils
import gwmemory
import matplotlib.pyplot as plt
import numpy as np

from .waveforms import wrap_at_maximum, apply_window, wrap_by_n_indices

gamma_lmlm = gwmemory.angles.load_gamma()

logger = bilby.core.utils.logger

roll_off = 0.2


def overlap_function(a, b, frequency, psd):
    psd_interp = psd.power_spectral_density_interpolated(frequency)
    duration = 1. / (frequency[1] - frequency[0])

    inner_a = utils.noise_weighted_inner_product(
        a['plus'], a['plus'], psd_interp, duration)

    inner_a += utils.noise_weighted_inner_product(
        a['cross'], a['cross'], psd_interp, duration)

    inner_b = utils.noise_weighted_inner_product(
        b['plus'], b['plus'], psd_interp, duration)

    inner_b += utils.noise_weighted_inner_product(
        b['cross'], b['cross'], psd_interp, duration)

    inner_ab = utils.noise_weighted_inner_product(
        a['plus'], b['plus'], psd_interp, duration)

    inner_ab += utils.noise_weighted_inner_product(
        a['cross'], b['cross'], psd_interp, duration)
    overlap = inner_ab / np.sqrt(inner_a * inner_b)
    return overlap.real


def wrap_by_time_shift(waveforms, time_shifts, time_per_index):
    index_shifts = np.round(time_shifts/time_per_index).astype(int)
    waveforms = np.roll(waveforms, shift=index_shifts)
    return waveforms


def time_domain_nr_hyb_sur_waveform_with_memory_arbitrary_wrapped_pp(memory_generator, inc, phases,
                                                                     time_shifts, **kwargs):
    times = memory_generator.times
    kwargs['alpha'] = 0.1

    waveforms_grid = [dict(plus=None, cross=None)] * len(time_shifts) * len(phases)
    waveforms = [dict(plus=None, cross=None)] * len(phases)

    for i in range(len(phases)):
        waveforms[i] = gwmemory.waveforms.combine_modes(memory_generator.h_lm, inc, phases[i])
        waveforms[i] = apply_window(waveform=waveforms[i], times=times, kwargs=kwargs)
        waveforms[i] = wrap_by_n_indices(shift=kwargs.get('shift'), waveform=waveforms[i])
        waveforms_grid[i] = waveforms[i]

    for i in range(len(waveforms), len(waveforms_grid)):
        waveforms_grid[i] = deepcopy(waveforms_grid[i % len(waveforms)])

    waveforms_plus = np.array([waveform['plus'] for waveform in waveforms_grid])
    waveforms_cross = np.array([waveform['plus'] for waveform in waveforms_grid])

    for j in range(len(phases)):
        for i in range(len(time_shifts)):
            waveforms_plus[i + j*len(time_shifts)] = \
                wrap_by_time_shift(waveforms=waveforms_plus[i + j*len(time_shifts)],
                                   time_shifts=time_shifts[i],
                                   time_per_index=(times[-1]-times[0])/len(times))
            waveforms_cross[i + j*len(time_shifts)] = \
                wrap_by_time_shift(waveforms=waveforms_cross[i + j*len(time_shifts)],
                                   time_shifts=time_shifts[i],
                                   time_per_index=(times[-1]-times[0])/len(times))

    for i in range(len(waveforms_grid)):
        waveforms_grid[i]['plus'] = waveforms_plus[i]
        waveforms_grid[i]['cross'] = waveforms_cross[i]

    frequency_array = None

    for i, waveform in enumerate(waveforms_grid):
        waveform['cross'], frequency_array = bilby.core.utils.nfft(waveform['cross'], memory_generator.sampling_frequency)
        waveform['plus'], _ = bilby.core.utils.nfft(waveform['plus'], memory_generator.sampling_frequency)
        waveforms_grid[i] = waveform

    return waveforms_grid, frequency_array


def time_domain_nr_hyb_sur_waveform_with_memory_arbitrary_wrapped_debug(memory_generator, inc, phase,
                                                                        time_shift, **kwargs):
    times = memory_generator.times
    waveform = gwmemory.waveforms.combine_modes(memory_generator.h_lm, inc, phase)
    waveform = apply_window(waveform=waveform, times=times, kwargs=kwargs)
    waveform = wrap_at_maximum(waveform=waveform, kwargs=kwargs)

    waveform['plus'] = wrap_by_time_shift(waveforms=waveform['plus'], time_shifts=time_shift,
                                          time_per_index=(times[-1]-times[0])/len(times))
    waveform['cross'] = wrap_by_time_shift(waveforms=waveform['cross'], time_shifts=time_shift,
                                           time_per_index=(times[-1]-times[0])/len(times))
    waveform['cross'], frequency_array = bilby.core.utils.nfft(waveform['cross'], memory_generator.sampling_frequency)
    waveform['plus'], _ = bilby.core.utils.nfft(waveform['plus'], memory_generator.sampling_frequency)

    return waveform, frequency_array


def adjust_phase_and_geocent_time(result, injection_model, recovery_model, ifo):
    parameters = result.posterior.iloc[-2].to_dict()
    print(parameters)
    # phase_grid_init = np.linspace(0, np.pi, 30)
    time_grid_init = np.linspace(-0.01, 0.00, 21)
    phase_grid_init = np.array([-0.7014326992696138, 0])
    phase_grid_init = np.array([0])
    # time_grid_init = np.array([-0.004363938949701662, 0, +0.004363938949701661])
    # time_grid_init = np.array([-6.931566285600532+6.92720234665083])
    # time_grid_init = np.array([0])
    # time_grid_init = np.array([-0.1, 0, 5])

    phase_grid_mesh, time_grid_mesh = np.meshgrid(phase_grid_init, time_grid_init)

    phase_grid = phase_grid_mesh.flatten()
    time_grid = time_grid_mesh.flatten()

    recovery_wg = bilby.gw.waveform_generator. \
        WaveformGenerator(frequency_domain_source_model=recovery_model,
                          duration=16, sampling_frequency=2048,
                          waveform_arguments=dict(alpha=0.1))

    memory_generator = gwmemory.waveforms.HybridSurrogate(q=parameters['mass_ratio'],
                                                          total_mass=parameters['total_mass'],
                                                          spin_1=parameters['s13'],
                                                          spin_2=parameters['s23'],
                                                          times=recovery_wg.time_array,
                                                          distance=parameters['luminosity_distance'],
                                                          minimum_frequency=10,
                                                          sampling_frequency=2048,
                                                          units='mks'
                                                          )
    wrap_check_wf = gwmemory.waveforms.combine_modes(memory_generator.h_lm, parameters['inc'], parameters['phase'])
    wrap_check_wf, shift = wrap_at_maximum(wrap_check_wf, dict())

    full_wf = recovery_wg.frequency_domain_strain(parameters)

    phases = (phase_grid_init + parameters['phase']) % (2 * np.pi)

    matching_wfs, frequency_array = time_domain_nr_hyb_sur_waveform_with_memory_arbitrary_wrapped_pp(
        memory_generator=memory_generator, inc=parameters['inc'],
        phases=phases, time_shifts=time_grid_init, shift=shift)

    # for matching_wf in matching_wfs:
    #     plt.xlim(20, 1000)
    #     plt.loglog()
    #     plt.plot(recovery_wg.frequency_array, np.abs(full_wf['plus']))
    #     plt.plot(frequency_array, np.abs(matching_wf['plus']))
    #     plt.show()
    #     plt.clf()
        # plt.xlim(20, 1000)
        # plt.loglog()
        # plt.plot(recovery_wg.frequency_array, np.abs(full_wf['cross']))
        # plt.plot(frequency_array, np.abs(matching_wf['cross']))
        # plt.plot(injected_wg.frequency_array, np.abs(test_wf['cross']))
        # plt.show()
        # plt.clf()

    overlaps = np.array([])
    for matching_wf in matching_wfs:
        overlaps = np.append(overlaps, overlap_function(full_wf, matching_wf, recovery_wg.frequency_array,
                                                        ifo.power_spectral_density))
    # plt.contourf(time_grid_mesh, phase_grid_mesh, np.reshape(overlaps, (len(time_grid_init), len(phase_grid_init))))
    plt.plot(overlaps)
    plt.xlabel('Time shift')
    plt.ylabel('Overlap')
    plt.show()
    plt.clf()

    overlaps = np.nan_to_num(overlaps)
    max_n0 = np.argmax(overlaps)
    time_shift = time_grid[max_n0]
    phase_shift = phase_grid[max_n0]
    print("Time shift:" + str(time_shift))
    print("Phase shift:" + str(phase_shift))



    new_result = deepcopy(result)
    for i in range(len(new_result.posterior['geocent_time'])):
        new_result.posterior.geocent_time.iloc[i] += time_shift
        new_result.posterior.phase.iloc[i] += phase_shift
        if new_result.posterior.phase.iloc[i] < 0:
            new_result.posterior.phase.iloc[i] += 2 * np.pi
        new_result.posterior.phase.iloc[i] %= 2 * np.pi
    # new_result.plot_corner(filename='post_processed_corner')
    return new_result
