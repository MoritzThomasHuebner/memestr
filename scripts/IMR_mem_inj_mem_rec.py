from __future__ import division
import memestr
import os

import submitter

outdir = submitter.create_fresh_numbered_outdir(outdir_base='IMR_mem_inj_mem_rec_')

memestr.wrappers.wrappers.run_basic_injection_imr_phenom(
    injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    outdir=outdir)

submitter.move_log_file_to_outdir(dir_path=os.path.dirname(os.path.realpath(__file__)),
                                  log_file='IMR_mem_inj_mem_rec.log',
                                  outdir=outdir)
