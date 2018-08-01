from __future__ import division
import memestr.core.waveforms as waveforms
from memestr.wrappers.wrappers import run_basic_injection

run_basic_injection(waveforms.time_domain_nr_sur_waveform_without_memory, 'outdir_non_memory')
