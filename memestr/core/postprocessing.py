import numpy as np
import bilby
from copy import deepcopy
logger = bilby.core.utils.logger


def overlap_function(a, b, ifo):
    inner_aa = ifo.inner_product(a)
    inner_bb = vec_inner_prod(b, b, ifo.power_spectral_density, ifo.strain_data.duration)
    inner_ab = vec_inner_prod(a, b, ifo.power_spectral_density, ifo.strain_data.duration)
    return np.real(inner_ab/np.sqrt(inner_aa*inner_bb))


def vec_inner_prod(aa, bb, power_spectral_density, duration):
    integrand = np.conj(aa) * bb / power_spectral_density
    return np.real(4 / duration * np.sum(integrand, axis=1))


def adjust_phase_and_geocent_time(result, injection_model, recovery_model, ifo):
    parameters = result.posterior.iloc[0]
    phase_grid = np.linspace(0, np.pi, 60) + parameters['phase']
    # dphi = (phase_grid[1] - phase_grid[0])/len(phase_grid)

    time_grid = np.linspace(-0.01, -0.003, 120)
    # dt_g = (time_grid[1]-time_grid[0])/len(time_grid)
    phase_grid, time_grid = np.meshgrid(phase_grid, time_grid)

    phase_grid = phase_grid.flatten()
    time_grid = time_grid.flatten()

    matching_wf = recovery_model(frequencies=ifo.frequency_array, **parameters)
    parameters_without_phase_and_time = deepcopy(parameters)
    del parameters_without_phase_and_time['geocent_time']
    del parameters_without_phase_and_time['phase']
    full_wf = injection_model(frequencies=ifo.frequency_array, phase=phase_grid,
                              geocent_time=time_grid, **parameters_without_phase_and_time)

    overlap = overlap_function(matching_wf, full_wf, ifo)

    max_n0 = np.argmax(overlap)
    time_shift = time_grid[max_n0]
    phase_new = phase_grid[max_n0]
    new_result = deepcopy(result)
    for i in len(new_result.posterior['geocent_time']):
        new_result.posterior.geocent_time.iloc[i] += time_shift
        new_result.posterior.phase.iloc[i] += phase_new
        if new_result.posterior.phase.iloc[i] < 0:
            new_result.posterior.phase.iloc[i] += np.pi
        new_result.posterior.phase.iloc[i] %= np.pi
    new_result.plot_corner(filename='post_processed_corner')
    return new_result
