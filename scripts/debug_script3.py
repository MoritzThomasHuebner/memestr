# import matplotlib

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from memestr.core.waveforms.phenom import *
from memestr.core.postprocessing import overlap_function
from bilby.gw.detector.psd import PowerSpectralDensity
from bilby.core.series import CoupledTimeAndFrequencySeries
from bilby.gw.waveform_generator import WaveformGenerator

# memestr.wrappers.injection_recovery.run_production_injection(
# injection_model=fd_nr_sur_with_memory,
# recovery_model='fd_nr_sur',
# filename_base=20021,
# outdir='20021_dynesty_production_IMR_non_mem_rec/', label='IMR_mem_inj_non_mem_rec',
# alpha=0.1, distance_marginalization=False,
# time_marginalization=True, phase_marginalization=True,
# sampler='dynesty', nthreads=1,
# npoints=200, duration=16, random_seed=42, dlogz=0.1,
# sampling_frequency=2048, resume=True, clean=False, n_check_point=100)
#

# memestr.routines.reweighting.run_reweighting(
#         injection_model=memestr.core.waveforms.td_imrd_with_memory,
#         recovery_model=memestr.core.waveforms.td_imrd_with_memory,
#         filename_base='20021_dynesty',
#         outdir='20021_dynesty_production_IMR_non_mem_rec/', label='IMR_mem_inj_non_mem_rec',
#         sub_run_id=10,
#         alpha=0.1, distance_marginalization=True,
#         time_marginalization=False, phase_marginalization=False,
#         sampler='dynesty', nthreads=1,
#         npoints=50, duration=16, random_seed=42, dlogz=0.1,
#         sampling_frequency=2048, resume=True, clean=False, n_check_point=100)
params = dict(total_mass=65, mass_ratio=0.8, luminosity_distance=1000.0,
              dec=-1.2108, ra=1.375, inc=1.5, psi=2.659,
              phase=1.3, geocent_time=4.0, s11=0.0, s12=0.0,
              s13=-0.23856274291657414, s21=0.0, s22=0.0, s23=-0.017810708471383503)
psd = PowerSpectralDensity.from_aligo()
series = CoupledTimeAndFrequencySeries(sampling_frequency=2048, duration=16)
wg_1 = WaveformGenerator(duration=series.duration, sampling_frequency=series.sampling_frequency,
                         frequency_domain_source_model=fd_imrd)
wg_2 = WaveformGenerator(duration=series.duration, sampling_frequency=series.sampling_frequency,
                         frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory_2)

wg_1.parameters = params
wg_2.parameters = params

print(overlap_function(wg_1.frequency_domain_strain(), wg_2.frequency_domain_strain(), series.frequency_array, psd))

plt.plot(wg_1.time_array, wg_1.time_domain_strain()['cross'])
plt.plot(wg_2.time_array, wg_2.time_domain_strain()['cross'])
plt.xlim(15.8, 16)
plt.show()
plt.clf()
