import tupak
import os

with open('distance_evidence.dat', 'a') as outfile:
    outfile.write("#Luminosity distance\tlog_bayes_factor\tlog_bayes_evidence\tlog_bayes_evidence_err")
run = "001_IMR_mem_inj_non_mem_rec"
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + run
for subdir, dirs, files in os.walk(dir_path):
    for dir in dirs:
        dir_path = subdir + "/" + dir + "/" + "IMRPhenomD_result.h5"
        print(dir_path)
        result = tupak.core.result.read_in_result(filename=dir_path)

        with open(run + '_distance_evidence.dat', 'a') as outfile:
            outfile.write(str(dir) + '\t' +
                          str(result.log_bayes_factor) + '\t' +
                          str(result.log_evidence) + '\t' +
                          str(result.log_evidence_err) + '\n')