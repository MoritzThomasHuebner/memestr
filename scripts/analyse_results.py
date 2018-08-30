import tupak
import os
run_id = '002'
runs = [run_id+'_IMR_mem_inj_mem_rec',
        run_id+'_IMR_mem_inj_non_mem_rec',
        run_id+'_IMR_non_mem_inj_mem_rec',
        run_id+'_IMR_non_mem_inj_non_mem_rec',
        run_id+'_IMR_pure_mem_inj_pure_mem_rec']

for run in runs:
    with open(run + '_distance_evidence.dat', 'a') as outfile:
        outfile.write("#Luminosity distance\tlog_bayes_factor\tlog_bayes_evidence\tlog_bayes_evidence_err")

    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + run
    for subdir, dirs, _ in os.walk(dir_path):

        dir_path = subdir + "/" + dirs[0] + "/" + "IMRPhenomD_result.h5"
        print(dir_path)

        result = tupak.core.result.read_in_result(filename=dir_path)
        with open(run + '_distance_evidence.dat', 'a') as outfile:
            outfile.write(str(dirs[0]) + '\t' +
                          str(result.log_evidence) + '\t' +
                          str(result.log_bayes_factor) + '\t' +
                          str(result.log_evidence_err) + '\n')