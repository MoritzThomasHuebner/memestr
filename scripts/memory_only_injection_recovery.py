from __future__ import division
import memestr

memestr.wrappers.wrappers.run_basic_injection(
    injection_model=memestr.core.waveforms.time_domain_nr_sur_memory_waveform,
    recovery_model=memestr.core.waveforms.time_domain_nr_sur_memory_waveform,
    outdir='outdir_memory_only_001')
