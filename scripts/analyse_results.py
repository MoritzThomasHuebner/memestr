import numpy as np
import bilby
import os
import sys


run_id = '016'
if sys.argv[1]:
    run_id = sys.argv[1]

runs = [run_id + '_IMR_mem_inj_mem_rec',
        run_id + '_IMR_mem_inj_non_mem_rec']

for run in runs:
    with open(run + '_distance_evidence.dat', 'w') as outfile:
        outfile.write("#Luminosity distance\tlog_bayes_factor\tlog_bayes_evidence\tlog_bayes_evidence_err\t"
                      "Network SNR\n")
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + run
        luminosity_distances = []
        log_evidence = []
        log_bayes_factor = []
        log_evidence_err = []
        network_snr = []
        for subdir, _, _ in os.walk(dir_path):
            files = os.listdir(subdir)

            for f in files:
                if 'result.h5' in f:
                    dir_path = subdir + "/" + f
                    print(dir_path)
                    result = bilby.core.result.read_in_result(filename=dir_path)
                    luminosity_distances.append(int(os.path.basename(subdir)))
                    log_evidence.append(result.log_evidence)
                    log_bayes_factor.append(result.log_bayes_factor)
                    try:
                        log_evidence_err.append(result.log_evidence_err)
                    except AttributeError:
                        log_evidence_err.append(0)
                    network_snr.append(np.sqrt(
                        np.sum(result.meta_data['likelihood'][detector]['optimal_SNR']**2)
                        for detector in ['H1', 'L1', 'V1']))

        data = zip(luminosity_distances, log_evidence, log_bayes_factor, log_evidence_err, network_snr)
        data = sorted(data)
        print(data)
        for row in data:
            outfile.write('{}\t{}\t{}\t{}\t{}\n'.format(*row))

# for run in runs:
#     with open(run + '_bf_max_like.dat', 'w') as outfile:
#         outfile.write("#log_bayes_factor\tMax likelihood\n")
#         dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + run
#         log_bayes_factor = []
#         max_likelihood = []
#         for subdir, _, _ in os.walk(dir_path):
#             files = os.listdir(subdir)
#
#             for f in files:
#                 if 'result.h5' in f:
#                     dir_path = subdir + "/" + f
#                     print(dir_path)
#                     result = bilby.core.result.read_in_result(filename=dir_path)
#
#                     log_bayes_factor.append(result.log_bayes_factor)
#                     max_likelihood.append(np.max(result.nested_samples.log_likelihood.values))
#
#         data = zip(log_bayes_factor, max_likelihood)
#         data = sorted(data)
#         for row in data:
#             outfile.write('{}\t{}\n'.format(*row))
