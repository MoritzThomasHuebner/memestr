import memestr.core
import memestr.routines

routines = dict(
    run_production_injection=memestr.routines.injection_recovery.run_production_injection,
    run_time_phase_optimization=memestr.routines.optimization.run_time_phase_optimization,
    run_reweighting=memestr.routines.reweighting.run_reweighting
)
