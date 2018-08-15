from __future__ import division
import memestr

memestr.wrappers.wrappers.run_basic_injection(
    injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    outdir='imr_debug')
