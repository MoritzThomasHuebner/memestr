import bilby as bb
import numpy as np
import corner

res = bb.result.read_in_result('9999_pypolychord_production_IMR_non_mem_rec/IMR_mem_inj_non_mem_rec_result.json')
res.plot_corner(parameters=['total_mass', 'mass_ratio', 'inc', 'luminosity_distance', 'ra', 'dec', 'psi',
                            's13', 's23', 'geocent_time', 'phase'])



file = '9999_pypolychord_production_IMR_non_mem_rec/IMR_mem_inj_non_mem_rec.txt'
weighted_samples = np.loadtxt(file)
filtered_samples = weighted_samples[:, -9:]
corner.corner(filtered_samples)
