import memestr.wrappers as wrappers
import memestr.core as core
import memestr.submit as submit

models = dict(
    time_domain_IMRPhenomD_memory_waveform=core.waveforms.time_domain_IMRPhenomD_memory_waveform,
    time_domain_IMRPhenomD_waveform_with_memory=core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    time_domain_IMRPhenomD_waveform_without_memory=core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
    time_domain_nr_sur_memory_waveform=core.waveforms.time_domain_nr_sur_memory_waveform,
    time_domain_nr_sur_waveform_with_memory=core.waveforms.time_domain_nr_sur_waveform_with_memory,
    time_domain_nr_sur_waveform_without_memory=core.waveforms.time_domain_nr_sur_waveform_without_memory,
    time_domain_nr_sur_waveform_with_memory_base_modes=core.waveforms.time_domain_nr_sur_waveform_with_memory_base_modes,
    time_domain_nr_sur_waveform_without_memory_base_modes=core.waveforms.time_domain_nr_sur_waveform_without_memory_base_modes
)

scripts = dict(
    run_basic_injection_imr_phenom=wrappers.wrappers.run_basic_injection_imr_phenom,
    run_basic_injection_nrsur=wrappers.wrappers.run_basic_injection_nrsur,
    run_basic_injection=wrappers.wrappers.run_basic_injection
)
