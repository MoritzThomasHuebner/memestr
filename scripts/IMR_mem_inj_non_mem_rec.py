from __future__ import division
import memestr
import os
import shutil

from . import submitter

outdir = submitter.create_fresh_numbered_outdir(outdir_base='IMR_mem_inj_non_mem_rec_')

memestr.wrappers.wrappers.run_basic_injection_imr_phenom(
    injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
    outdir=outdir)

log_file = 'IMR_mem_inj_non_mem_rec.log'
dir_path = os.path.dirname(os.path.realpath(__file__))
os.rename(dir_path + log_file, dir_path + outdir + "/" + log_file)
shutil.move(dir_path + log_file, dir_path + outdir + "/" + log_file)