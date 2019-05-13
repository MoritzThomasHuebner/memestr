from copy import deepcopy

from scipy.interpolate import CubicSpline

import bilby
import gwmemory

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


def wrap_by_time_shift_continuous(times, waveforms, time_shifts):
    waveform_interpolants = CubicSpline(times, waveforms, extrapolate='periodic')
    new_times = (times + time_shifts) % np.max(times)
    return waveform_interpolants(new_times)


def time_domain_nr_hyb_sur_waveform_with_memory_arbitrary_wrapped_pp(memory_generator, inc, phases,
                                                                     time_shifts, **kwargs):
    times = memory_generator.times
    kwargs['alpha'] = 0.1

    waveforms_grid = [dict(plus=None, cross=None)] * len(time_shifts) * len(phases)
    waveforms = [dict(plus=None, cross=None)] * len(phases)

    # generate basic waveform
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
                wrap_by_time_shift(waveforms=waveforms_plus[i + j * len(time_shifts)],
                                   time_shifts=time_shifts[i],
                                   time_per_index=(times[-1]-times[0])/len(times))
            waveforms_cross[i + j*len(time_shifts)] = \
                wrap_by_time_shift(waveforms=waveforms_cross[i + j * len(time_shifts)],
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
    time_grid_init = np.linspace(-0.008, -0.002, 301)
    phase_grid_init = np.array([-0.7014326992696138])
    # phase_grid_init = np.array([0])
    # time_grid_init = np.array([-0.004363938949701662, 0, +0.004363938949701661])
    # time_grid_init = np.array([-6.9315664109380615+6.92720234665083])
    # time_grid_init = np.array([0])
    # time_grid_init = np.array([-0.1, 0, 5])

    phase_grid_mesh, time_grid_mesh = np.meshgrid(phase_grid_init, time_grid_init)

    phase_grid = phase_grid_mesh.flatten()
    time_grid = time_grid_mesh.flatten()

    recovery_wg = bilby.gw.waveform_generator. \
        WaveformGenerator(frequency_domain_source_model=recovery_model,
                          duration=16, sampling_frequency=8192,
                          waveform_arguments=dict(alpha=0.1))

    memory_generator = gwmemory.waveforms.HybridSurrogate(q=parameters['mass_ratio'],
                                                          total_mass=parameters['total_mass'],
                                                          spin_1=parameters['s13'],
                                                          spin_2=parameters['s23'],
                                                          times=recovery_wg.time_array,
                                                          distance=parameters['luminosity_distance'],
                                                          minimum_frequency=10,
                                                          sampling_frequency=8192,
                                                          units='mks',
                                                          # l_max=2
                                                          )
    # print(memory_generator.modes)
    # homs = [(4, -3), (3, 2), (4, -4), (3, 3), (3, 0), (4, 4), (3, 1),  (2, 1), (3, -2), (3, -1),  (2, -1),
    #      (4, 3), (4, 2), (3, -3), (4, -2), (2, -2)]
    # (2, -2),(2, 0),(2, 2),
    # for hom in homs:
    #     del memory_generator.h_lm[hom]
    # print(memory_generator.h_lm.keys())
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

    overlaps = np.nan_to_num(overlaps)
    max_n0 = np.argmax(overlaps)
    print('Maximum overlap: ' + str(overlaps[max_n0]))
    time_shift = time_grid[max_n0]
    phase_shift = phase_grid[max_n0]
    print("Time shift:" + str(time_shift))
    print("Phase shift:" + str(phase_shift))

    # plt.contourf(time_grid_mesh, phase_grid_mesh, np.reshape(overlaps, (len(time_grid_init), len(phase_grid_init))))
    plt.plot(time_grid, overlaps)
    plt.axvline(time_grid[max_n0])
    # plt.axvline(time_grid[max_n0])
    # plt.axvline(-0.7014326992696138 + np.pi)
    plt.xlabel('Time shift')
    plt.ylabel('Overlap')
    # plt.xlabel('Time shift')
    # plt.ylabel('Phase shift')
    # plt.colorbar()
    # plt.title('Overlap')
    plt.show()
    plt.clf()

    new_result = deepcopy(result)
    for i in range(len(new_result.posterior['geocent_time'])):
        new_result.posterior.geocent_time.iloc[i] += time_shift
        new_result.posterior.phase.iloc[i] += phase_shift
        if new_result.posterior.phase.iloc[i] < 0:
            new_result.posterior.phase.iloc[i] += 2 * np.pi
        new_result.posterior.phase.iloc[i] %= 2 * np.pi
    return new_result


import lalsimulation as lalsim
import numpy as np
import bilby.gw.utils as utils
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import bilby
import gwsurrogate as gws

MASS_TO_TIME = 4.925491025543576e-06  # solar masses to seconds
MASS_TO_DISTANCE = 4.785415917274702e-20  # solar masses to Mpc
MPC = 3.08568e22  # Mpc in metres
MSUN = 1.98855e30  # solar mass in  kg

# Evaluate the NRHybSur waveform
sur = gws.LoadSurrogate('NRHybSur3dq8')

def vec_inner_prod(aa, bb, power_spectral_density, duration):
    integrand = np.conj(aa) * bb / power_spectral_density
    return np.real(4 / duration * np.sum(integrand, axis=1))

def convert_time_strain_to_frequency(h, t, time,
        sampling_frequency, minimum_frequency, frequency):

    #h = - np.real(h) + 1j * np.imag(h)
    h = interp1d(t, h, bounds_error=False, fill_value=0.0)(time)
    h = h *  tukey(len(h),alpha=0.1)

    plus_t = np.real(h); cross_t = -np.imag(h)

    plus = np.fft.rfft(plus_t) / sampling_frequency
    cross = np.fft.rfft(cross_t) / sampling_frequency

    plus[frequency < minimum_frequency] = complex(0.0)
    cross[frequency < minimum_frequency] = complex(0.0)

    return plus, cross

def convert_time_strain_to_frequency_multiple(
        h, t, time,
        sampling_frequency, minimum_frequency, frequency):

    #h = -np.real(h) + 1j * np.imag(h)

    h = interp1d(t, h, bounds_error=False, fill_value=0.0, axis=1)(time)
    h = (h*tukey(len(h[0]),alpha=0.1).T)

    plus_t = np.real(h); cross_t = -np.imag(h)

    plus = np.fft.rfft(plus_t) / sampling_frequency
    cross = np.fft.rfft(cross_t) / sampling_frequency

    plus[:, frequency < minimum_frequency] = complex(0.0)
    cross[:, frequency < minimum_frequency] = complex(0.0)

    return plus, cross

def fun_overlap(ap, ac, bp, bc, frequency):
    PSD = bilby.gw.detector.PowerSpectralDensity(asd_file='aLIGO_ZERO_DET_high_P_asd.txt')
    psd_interp = PSD.power_spectral_density_interpolated(frequency)
    duration = 1./(frequency[1]-frequency[0])

    inner_ap = utils.noise_weighted_inner_product(ap, ap,
                psd_interp, duration)
    inner_ac = utils.noise_weighted_inner_product(ac, ac,
                psd_interp, duration)

    inner_bp = vec_inner_prod(bp, bp,
                psd_interp, duration)
    inner_bc = vec_inner_prod(bc, bc,
                psd_interp, duration)

    inner_abp = vec_inner_prod(ap, bp, psd_interp, duration)
    inner_abc = vec_inner_prod(ac, bc, psd_interp, duration)

    return np.real((inner_abp+inner_abc)/np.sqrt((inner_ac+inner_ap)*(inner_bc+inner_bp)))

def big_O(a, b, frequency):
    PSD = bilby.gw.detector.PowerSpectralDensity(asd_file='aLIGO_ZERO_DET_high_P_asd.txt')
    psd_interp = PSD.power_spectral_density_interpolated(frequency)
    duration = 1./(frequency[1]-frequency[0])

    inner_a = utils.noise_weighted_inner_product(a, a,
                psd_interp, duration)

    inner_b = utils.noise_weighted_inner_product(b, b,
                psd_interp, duration)

    inner_ab = utils.noise_weighted_inner_product(a, b,
                psd_interp, duration)

    return np.real(inner_ab/np.sqrt(inner_a*inner_b))

# ==============================================================================
# Waveform function definitions
# ==============================================================================

def gws_nominal(frequency, mass_1, mass_2, luminosity_distance, chi_1,
                      chi_2, theta_jn, phase, psi,
                      geocent_time, ra, dec, **kwargs):

    # Create the waveform function arguments
    waveform_kwargs = dict(reference_frequency=50.0, minimum_frequency=20.0)
    waveform_kwargs.update(kwargs)
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']

    # sampling parameters
    duration = 1./(frequency[1]-frequency[0])
    dt = 1./(2*frequency[-1])
    time = np.arange(0.0, duration, dt)
    sampling_frequency = frequency[-1]*2.0

    spin_1x = spin_1y = spin_2x = spin_2y = 0.0
    spin_1z = chi_1; spin_2z = chi_2

    # transformation of mass parameters
    q = mass_1/mass_2
    MTot = mass_1 + mass_2

    # create the spin vectors
    S1 = np.array([spin_1x, spin_1y, spin_1z])
    S2 = np.array([spin_2x, spin_2y, spin_2z])

    # Create surrogate
    x = [q, S1[2], S2[2]]
    modes_full = [(2,2), (2,1), (2,0), (3,3), (3,2),
                  (3,1), (3,0), (4,4), (4,3), (4,2), (5,5)]

    # setting up the time domain
    epsilon = 100*MASS_TO_TIME*MTot
    t_NR = np.arange(-duration/1.3+epsilon, epsilon, dt)

    h = sur(x, times=t_NR, f_low=0, M=MTot,
            dist_mpc=luminosity_distance, units='mks', f_ref=reference_frequency)
    t_NR -= t_NR[0]

    h_NR = np.zeros(len(h[modes_full[0]]), dtype=complex)
    for mode in modes_full:
        h_NR += (gwmemory.harmonics.sYlm(-2, mode[0], mode[1],
                     theta_jn, phase+np.pi/2)*
                     h[mode])

        if mode[1] > 0:
            h_NR += (gwmemory.harmonics.sYlm(-2, mode[0], -mode[1],
                         theta_jn, phase+np.pi/2)*(-1)**mode[0]*
                         np.conj(h[mode]))

        if mode == (2,2):
            y22_time_shift = t_NR[np.argmax(h[(2,2)])]

    plus, cross = convert_time_strain_to_frequency(h_NR, t_NR, time,
                        sampling_frequency, minimum_frequency, frequency)

    plus = plus* np.exp(-2j * np.pi * (duration - y22_time_shift) * frequency)
    cross = cross* np.exp(-2j * np.pi * (duration - y22_time_shift) * frequency)

    return {'plus' : plus, 'cross' : cross}

def gws_overlap(frequency, mass_1, mass_2, luminosity_distance,
                chi_1, chi_2, theta_jn, phase, psi,
                geocent_time, ra, dec, **kwargs):

    # Create the waveform function arguments
    waveform_kwargs = dict(reference_frequency=50.0, minimum_frequency=20.0,
                           comparison_waveform='IMRPhenomPv2', return_correction=False)
    waveform_kwargs.update(kwargs)
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    comparison_waveform = waveform_kwargs['comparison_waveform']
    return_correction = waveform_kwargs['return_correction']

    # sampling parameters
    duration = 1./(frequency[1]-frequency[0])
    dt = 1./(2*frequency[-1])
    time = np.arange(0.0, duration, dt)
    sampling_frequency = frequency[-1]*2.0
    delta_frequency = 1./duration

    # transformation of mass parameters
    q = mass_1/mass_2
    MTot = mass_1 + mass_2

    # create the spin vectors
    S1 = np.array([0.0, 0.0, chi_1])
    S2 = np.array([0.0, 0.0, chi_2])

    # Compute the comparison
    approximant = lalsim.GetApproximantFromString(comparison_waveform)
    data = lalsim.SimInspiralChooseFDWaveform(
        mass_1 * MSUN, mass_2 * MSUN,
        S1[0], S1[1], S1[2], S2[0], S2[1], S2[2],
        luminosity_distance * MPC,
        theta_jn, phase, 0.0, 0.0, 0.0,
        delta_frequency, minimum_frequency, sampling_frequency/2.,
        reference_frequency, None, approximant)

    h_comp = (data[0].data.data + 1j * data[1].data.data)

    # Create surrogate
    x = [q, S1[2], S2[2]]
    modes_full = [(2,2), (2,1), (2,0), (3,3), (3,2),
                  (3,1), (3,0), (4,4), (4,3), (4,2), (5,5)]

    # setting up the time domain
    epsilon = 100*MASS_TO_TIME*MTot
    t_NR = np.arange(-duration/1.3+epsilon, epsilon, dt)

    h = sur(x, times=t_NR, f_low=0, M=MTot,
            dist_mpc=luminosity_distance, units='mks', f_ref=reference_frequency)
    t_NR -= t_NR[0]


    # generate the grid to optimize over
    if duration == 4.0: grid_size = 100
    elif duration == 8.0: grid_size = 80
    elif duration == 16.0: grid_size = 60
    else:
        print('unknown duration')

    phase_grid = np.linspace(-np.pi/2, np.pi/2, grid_size) + phase # np.linspace(1.06,1.07,13) #
    time_grid = np.linspace(-50, 0.0, grid_size) * MASS_TO_TIME * MTot
    #np.linspace(-0.020, 0.000, 120) #+ duration/1.1 # [-0.009] #
    phase_grid, time_grid = np.meshgrid(phase_grid, time_grid)
    phase_grid = phase_grid.flatten(); time_grid = time_grid.flatten()

    # Create the 22 waveform
    h_NR = np.zeros((len(phase_grid), len(h[modes_full[0]])), dtype=complex)
    for mode in modes_full:
        h_NR +=  np.outer(gwmemory.harmonics.sYlm(-2, mode[0], mode[1],
                     theta_jn, phase_grid+np.pi/2),
                     h[mode])
        if mode[1] > 0:
            h_NR +=  np.outer(gwmemory.harmonics.sYlm(-2, mode[0], -mode[1],
                         theta_jn, phase_grid+np.pi/2)*(-1)**mode[0],
                         np.conj(h[mode]))

    y22_time_shift = t_NR[np.argmax(h[(2,2)])]

    plus, cross = convert_time_strain_to_frequency_multiple(h_NR, t_NR, time,
        sampling_frequency, minimum_frequency, frequency)


    plus = plus* np.exp(-2j * np.pi * np.outer(time_grid + duration - y22_time_shift, frequency))
    cross = cross* np.exp(-2j * np.pi * np.outer(time_grid + duration - y22_time_shift, frequency))

    h22 = plus+1j*cross

    overlap = fun_overlap(data[0].data.data, data[1].data.data, bp=plus, bc=cross, frequency=frequency)

    max_nO = np.argmax(overlap)
    max_O = np.max(overlap)
    time_shift = time_grid[max_nO]
    phase_new = phase_grid[max_nO]

    print('time shift: {}'.format(time_shift))
    print('new phase: {}'.format(phase_new))
    print('overlap max: {}'.format(overlap[max_nO]))

    plt.clf()
    plt.scatter(phase_grid, time_grid, c=overlap)
    plt.colorbar()
    plt.scatter([phase_new], [time_shift], c='r', alpha=0.5)
    plt.savefig('test.png')
    plt.clf()

    h_NR = np.zeros(len(h[modes_full[0]]), dtype=complex)
    for mode in modes_full:
        h_NR += (gwmemory.harmonics.sYlm(-2, mode[0], mode[1],
                     theta_jn, phase_new+np.pi/2)*
                     h[mode])
        if mode[1] > 0:
            h_NR += (gwmemory.harmonics.sYlm(-2, mode[0], -mode[1],
                         theta_jn, phase_new+np.pi/2)*(-1)**mode[0]*
                         np.conj(h[mode]))

    plus, cross = convert_time_strain_to_frequency(h_NR, t_NR, time,
                        sampling_frequency, minimum_frequency, frequency)

    plus = plus* np.exp(-2j * np.pi * (time_shift+duration - y22_time_shift) * frequency)
    cross = cross* np.exp(-2j * np.pi * (time_shift+duration - y22_time_shift) * frequency)

    h_NR = plus +1j*cross

    plt.clf()
    plt.loglog(frequency, np.abs(h_NR))
    plt.loglog(frequency, np.abs(h22[max_nO]))
    plt.loglog(frequency, np.abs(h_comp))
    plt.xlim(10,700)
    plt.savefig('test2.png')
    plt.clf()

    plt.semilogx(frequency, np.angle(h_NR))
    plt.semilogx(frequency, np.angle(h22[max_nO]))
    plt.semilogx(frequency, np.angle(h_comp))
    plt.xlim(10,700)
    plt.savefig('test3.png')
    plt.clf()

    print('true overlap: {}'.format(big_O(h22[max_nO], h_NR, frequency)))
    if return_correction:
        return {'plus' : plus, 'cross' : cross, 't_shift': time_shift,
                'phase_new' : phase_new, 'overlap': max_O}

    else:
        return {'plus' : plus, 'cross' : cross}
