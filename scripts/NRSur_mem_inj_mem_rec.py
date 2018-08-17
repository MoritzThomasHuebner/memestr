from __future__ import division
import memestr

from . import submitter

outdir = submitter.create_fresh_numbered_outdir(outdir_base='NRSur_mem_inj_mem_rec_')

memestr.wrappers.wrappers.run_basic_injection(
    injection_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
    outdir=outdir)
