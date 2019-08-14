import memestr.core.waveforms.surrogate
import memestr.routines as wrappers
import memestr.core as core
import memestr.routines.optimization
import memestr.routines.reweighting

models = dict(
    frequency_domain_IMRPhenomD_waveform_without_memory=core.waveforms.frequency_domain_IMRPhenomD_waveform_without_memory
)

scripts = dict(
    run_production_injection_imr_phenom=routines.injection_recovery.run_production_injection_imr_phenom,
    run_time_phase_optimization=memestr.routines.optimization.run_time_phase_optimization,
    run_reweighting=memestr.routines.reweighting.run_reweighting
)
