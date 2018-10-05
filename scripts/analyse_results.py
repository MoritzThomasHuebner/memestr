import numpy as np
import bilby
import os

run_id = '011'
runs = [run_id + '_IMR_mem_inj_mem_rec',
        run_id + '_IMR_mem_inj_non_mem_rec',
        run_id + '_IMR_non_mem_inj_mem_rec',
        run_id + '_IMR_non_mem_inj_non_mem_rec',
        run_id + '_IMR_pure_mem_inj_pure_mem_rec']

for run in runs:
    with open(run + '_distance_evidence.dat', 'w') as outfile:
        outfile.write("#Luminosity distance\tlog_bayes_factor\tlog_bayes_evidence\tlog_bayes_evidence_err\n")
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + run
        luminosity_distances = []
        log_evidence = []
        log_bayes_factor = []
        log_evidence_err = []
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

        data = zip(luminosity_distances, log_evidence, log_bayes_factor, log_evidence_err)
        data.sort()
        print(data)
        for row in data:
            outfile.write('{}\t{}\t{}\t{}\n'.format(*row))
