from memestr.submit import submitter
from memestr.wrappers import wrappers
from memestr import models, scripts

submitter.create_injection_parameter_set(10, wrappers.sample_injection_parameters)
params = submitter.get_injection_bash_strings(5)
print(params)
#  memestr.wrappers.wrappers.run_basic_injection_imr_phenom(injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
#                                                          recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
#                                                          outdir='test')
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
