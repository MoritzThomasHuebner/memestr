from __future__ import division
import memestr
import os

import submitter

naming_scheme = 'NRSur_mem_inj_mem_rec'
outdir = submitter.create_fresh_numbered_outdir(outdir_base=naming_scheme)

memestr.wrappers.wrappers.run_basic_injection(
    injection_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
    outdir=outdir)

submitter.move_log_file_to_outdir(dir_path=os.path.dirname(os.path.realpath(__file__)),
                                  log_file=naming_scheme+'.log',
                                  outdir=outdir)
