from __future__ import division
import memestr

memestr.wrappers.wrappers.run_basic_injection_imr_phenom(
    injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    outdir='IMR_mem_inj_mem_rec_001')
