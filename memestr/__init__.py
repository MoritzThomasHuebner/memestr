import wrappers
import core
import submit

models = dict(
    time_domain_IMRPhenomD_memory_waveform=core.waveforms.time_domain_IMRPhenomD_memory_waveform,
    time_domain_IMRPhenomD_waveform_with_memory=core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    time_domain_IMRPhenomD_waveform_without_memory=core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
    time_domain_nr_sur_memory_waveform=core.waveforms.time_domain_nr_sur_memory_waveform,
    time_domain_nr_sur_waveform_with_memory=core.waveforms.time_domain_nr_sur_waveform_with_memory,
    time_domain_nr_sur_waveform_without_memory=core.waveforms.time_domain_nr_sur_waveform_without_memory
)
scripts = dict(
    run_basic_injection_imr_phenom=wrappers.wrappers.run_basic_injection_imr_phenom,
    run_basic_injection=wrappers.wrappers.run_basic_injection
)