from __future__ import division
import sys
import os
import numpy as np

from memestr.submit import submitter
from memestr import models, scripts


outdir = sys.argv[1]
script = scripts[sys.argv[2]]
injection_model = models[sys.argv[3]]
recovery_model = models[sys.argv[4]]
number_of_tasks = int(sys.argv[5])

# outdir = 'debug_mp'
# script = scripts['run_basic_injection_imr_phenom']
# injection_model = models['time_domain_IMRPhenomD_waveform_without_memory']
# recovery_model = models['time_domain_IMRPhenomD_waveform_without_memory']
# number_of_tasks = 2

kwargs = submitter.parse_kwargs(input=sys.argv[5:])

dir_path = os.path.dirname(os.path.realpath(__file__))
result = submitter.run_job(outdir=outdir,
                           script=script,
                           dir_path=dir_path,
                           injection_model=injection_model,
                           recovery_model=recovery_model,
                           **kwargs)

with open(str(outdir[0:3]) + '_distance_evidence.dat', 'w') as outfile:
    if not list(outfile):
        outfile.write("#Luminosity distance\tlog_bayes_factor\tlog_bayes_evidence\tlog_bayes_evidence_err\n")
    luminosity_distance = result.injection_parameters['luminosity_distance']
    log_evidence = result.log_evidence
    log_bayes_factor = result.log_bayes_factor
    try:
        log_evidence_err = result.log_evidence_err
    except AttributeError:
        log_evidence_err = 0
    data = zip(luminosity_distance, log_evidence, log_bayes_factor, log_evidence_err)
    outfile.write('{}\t{}\t{}\t{}\n'.format(luminosity_distance, log_evidence, log_bayes_factor, log_evidence_err))

with open(str(outdir[0:3]) + '_bf_max_like.dat', 'w') as outfile:
    if not list(outfile):
        outfile.write("#log_bayes_factor\tMax likelihood\n")
    log_bayes_factor = result.log_bayes_factor
    max_likelihood = np.max(result.nested_samples.log_likelihood.values)
    outfile.write('{}\t{}\n'.format(log_bayes_factor, log_evidence_err))
