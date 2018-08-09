from __future__ import division
import memestr

memestr.wrappers.wrappers.run_basic_injection(
    injection_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_nr_sur_waveform_without_memory,
    outdir='outdir_non_memory_7')
