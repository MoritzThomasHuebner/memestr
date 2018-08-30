import tupak
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + "001_IMR_mem_inj_mem_rec"
for subdir, dirs, files in os.walk(dir_path):
    for dir in dirs:
        dir_path = subdir + "/" + dir + "/" + "IMRPhenomD_result.h5"
        print(dir_path)
        result = tupak.core.result.read_in_result(filename=dir_path)

        with open('distance_evidence.dat', 'a') as outfile:
            outfile.write(str(subdir) + '\t' +
                          str(result.bayes_factor) + '\t' +
                          str(result.bayes_factor_err) + '\t' +
                          str(result.log_evidence) + '\t' +
                          str(result.log_evidence_err) + '\n')