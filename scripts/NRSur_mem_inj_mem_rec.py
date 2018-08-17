from __future__ import division
import memestr
import os
import shutil

from . import submitter

outdir = submitter.create_fresh_numbered_outdir(outdir_base='NRSur_mem_inj_mem_rec_')

memestr.wrappers.wrappers.run_basic_injection(
    injection_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
    outdir=outdir)

log_file = 'NRSur_mem_inj_mem_rec.log'
dir_path = os.path.dirname(os.path.realpath(__file__))
os.rename(dir_path + log_file, dir_path + outdir + "/" + log_file)
shutil.move(dir_path + log_file, dir_path + outdir + "/" + log_file)