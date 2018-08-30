from memestr.submit import submitter
from memestr import models, scripts

# memestr.wrappers.wrappers.run_basic_injection_imr_phenom(injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
#                                                          recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
#                                                          outdir='test')
submitter.run_job(outdir='NRSur_Base_Mode',
                  script=scripts['run_basic_injection_nrsur'],
                  injection_model=models['time_domain_nr_sur_waveform_without_memory'],
                  recovery_model=models['time_domain_nr_sur_waveform_without_memory'],
                  luminosity_distance=50,
                  l_max=2,
                  mass_ratio=1.2414,
                  total_mass=65,
                  npoints=500)
