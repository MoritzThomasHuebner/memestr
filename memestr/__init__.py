import memestr.core
import memestr.routines

scripts = dict(
    run_production_injection=memestr.scripts.injection_recovery.run_production_injection,
    run_time_phase_optimization=memestr.scripts.optimization.run_time_phase_optimization,
    run_reweighting=memestr.scripts.reweighting.run_reweighting
)
