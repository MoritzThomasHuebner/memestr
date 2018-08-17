from __future__ import division
import memestr
import os

import submitter

submitter.run_job(naming_scheme='NRSur_mem_inj_mem_rec',
                  script=memestr.wrappers.wrappers.run_basic_injection,
                  injection_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory,
                  recovery_model=memestr.core.waveforms.time_domain_nr_sur_waveform_with_memory)