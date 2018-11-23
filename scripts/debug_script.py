import memestr
import bilby
from memestr.submit import submitter
from memestr.core import waveforms
from memestr.wrappers import wrappers
from memestr import models, scripts
import matplotlib.pyplot as plt
import numpy as np

# Debug Waveform plots
# submitter.create_injection_parameter_set(256, wrappers.sample_injection_parameters)
# times = np.linspace(0, 4, 32000)
# for i in range(16):
#     params = submitter.get_injection_parameter_set(i)
#     params['inc'] = params['iota']
#     del params['iota']
#     print(params)
#     a = waveforms.time_domain_IMRPhenomD_waveform_with_memory(times, **params)
#     b = waveforms.time_domain_IMRPhenomD_waveform_without_memory(times, **params)
#     c = waveforms.time_domain_IMRPhenomD_memory_waveform(times, **params)
#     # plt.plot(a['plus'])
#     # plt.plot(b['plus'])
#     plt.plot(c['plus'])
#     plt.show()
#     plt.clf()

# memestr.wrappers.wrappers.run_basic_injection_imr_phenom(injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
#                                                          recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
#                                                          outdir='test', sample='auto', npoints=300, walks=45)
# memestr.wrappers.wrappers.run_basic_injection_nrsur(injection_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
#                                                     recovery_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
#                                                     outdir='test', sampler='dynesty', npoints=1000)
# submitter.run_job(outdir='debug',
#                   script=scripts['run_basic_injection_imr_phenom'],
#                   injection_model=models['time_domain_IMRPhenomD_waveform_without_memory'],
#                   recovery_model=models['time_domain_IMRPhenomD_waveform_without_memory'],
#                   luminosity_distance=50,
#                   l_max=4,
#                   mass_ratio=1.2414,
#                   total_mass=65,
#                   npoints=500,
#                   iota=0.4,
#                   psi=2.659,
#                   phase=1.3,
#                   geocent_time=1126259642.413,
#                   ra=1.375,
#                   dec=-1.2108,
#                   random_seed=3,
#                   time_marginalization=True,
#                   zero_noise=False,
#                   random_injection_parameters=True,
#                   label='IMR_non_mem_inj_mem_rec')

# submitter.run_job(outdir='NRSur_HOM_IMR_Base_Mode',
#                   script=scripts['run_basic_injection_nrsur'],
#                   injection_model=models['time_domain_nr_sur_waveform_with_memory'],
#                   recovery_model=models['time_domain_nr_sur_waveform_with_memory'],
#                   luminosity_distance=50,
#                   l_max=4,
#                   mass_ratio=1.2414,
#                   total_mass=65,
#                   npoints=5000,
#                   iota=0.4,
#                   psi=2.659,
#                   phase=1.3,
#                   geocent_time=1126259642.413,
#                   ra=1.375,
#                   dec=-1.2108)
