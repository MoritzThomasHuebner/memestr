import memestr.core
import memestr.core.waveforms.surrogate
import memestr.core.waveforms.phenom
import memestr.routines as wrappers
import memestr.routines.optimization
import memestr.routines.reweighting

routines = dict(
    run_production_injection=routines.injection_recovery.run_production_injection,
    run_time_phase_optimization=memestr.routines.optimization.run_time_phase_optimization,
    run_reweighting=memestr.routines.reweighting.run_reweighting
)
