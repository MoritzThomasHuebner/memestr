from __future__ import division
import multiprocessing as mp
import sys
from memestr.submit import submitter
from memestr import models, scripts
import os
import numpy as np


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

dir_path = os.path.dirname(os.path.realpath(__file__))
kwargs = dict()
for arg in sys.argv[6:]:
    print(arg)
    key = arg.split("=")[0]
    value = arg.split("=")[1]
    if any(char.isdigit() for char in value):
        if all(char.isdigit() for char in value):
            value = int(value)
        else:
            value = float(value)
    kwargs[key] = value
print(kwargs)


# Define an output queue
output = mp.Queue()

# Setup a list of processes that we want to run
processes = [mp.Process(target=submitter.run_job,
                        args=(output, outdir + '/' + str(x), script),
                        kwargs=dict(injection_model=injection_model,
                                    recovery_model=recovery_model,
                                    **kwargs)) for x in range(number_of_tasks)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]


with open(str(outdir[0:3]) + '_distance_evidence.dat', 'w') as outfile:
    outfile.write("#Luminosity distance\tlog_bayes_factor\tlog_bayes_evidence\tlog_bayes_evidence_err\n")
    luminosity_distances = []
    log_evidence = []
    log_bayes_factor = []
    log_evidence_err = []
    for result in results:
        luminosity_distances.append(result.injection_parameters['luminosity_distance'])
        log_evidence.append(result.log_evidence)
        log_bayes_factor.append(result.log_bayes_factor)
        try:
            log_evidence_err.append(result.log_evidence_err)
        except AttributeError:
            log_evidence_err.append(0)
    data = zip(luminosity_distances, log_evidence, log_bayes_factor, log_evidence_err)
    data = sorted(data)
    print(data)
    for row in data:
        outfile.write('{}\t{}\t{}\t{}\n'.format(*row))

with open(str(outdir[0:3]) + '_bf_max_like.dat', 'w') as outfile:
    outfile.write("#log_bayes_factor\tMax likelihood\n")
    log_bayes_factor = []
    max_likelihood = []
    for result in results:
        log_bayes_factor.append(result.log_bayes_factor)
        max_likelihood.append(np.max(result.nested_samples.log_likelihood.values))
    data = zip(log_bayes_factor, max_likelihood)
    data = sorted(data)
    for row in data:
        outfile.write('{}\t{}\n'.format(*row))
