import memestr.core.waveforms.surrogate
import memestr.wrappers as wrappers
import memestr.core as core

models = dict(
    frequency_domain_IMRPhenomD_waveform_without_memory=core.waveforms.frequency_domain_IMRPhenomD_waveform_without_memory
)

scripts = dict(
    run_production_injection_imr_phenom=wrappers.injection_recovery.run_production_injection_imr_phenom,
    run_time_phase_optimization=wrappers.injection_recovery.run_time_phase_optimization,
    run_reweighting=wrappers.injection_recovery.run_reweighting,
    run_basic_injection_imr_phenom=wrappers.injection_recovery.run_basic_injection_imr_phenom,
)
