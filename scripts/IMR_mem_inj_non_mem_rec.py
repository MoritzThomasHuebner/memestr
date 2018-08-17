from __future__ import division
import memestr

from scripts import submitter

outdir = submitter.create_fresh_numbered_outdir(outdir_base='IMR_mem_inj_non_mem_rec_')

memestr.wrappers.wrappers.run_basic_injection_imr_phenom(
    injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
    outdir=outdir)
