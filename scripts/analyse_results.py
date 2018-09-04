import tupak
import os
run_id = '003'
runs = [run_id+'_IMR_mem_inj_mem_rec',
        run_id+'_IMR_mem_inj_non_mem_rec',
        run_id+'_IMR_non_mem_inj_mem_rec',
        run_id+'_IMR_non_mem_inj_non_mem_rec',
        run_id+'_IMR_pure_mem_inj_pure_mem_rec']

for run in runs:
    with open(run + '_distance_evidence.dat', 'w') as outfile:
        outfile.write("#Luminosity distance\tlog_bayes_factor\tlog_bayes_evidence\tlog_bayes_evidence_err\n")
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + run
        for subdir, dirs, _ in os.walk(dir_path):
            for dir in dirs:
                dir_path = subdir + "/" + dir + "/" + "IMRPhenomD_result.h5"
                print(dir_path)
                try:
                    result = tupak.core.result.read_in_result(filename=dir_path)
                except ValueError:
                    break
                outfile.write(str(dir) + '\t' +
                              str(result.log_evidence) + '\t' +
                              str(result.log_bayes_factor) + '\t' +
                              str(result.log_evidence_err) + '\n')
