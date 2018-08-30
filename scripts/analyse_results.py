import tupak
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + "001_IMR_mem_inj_mem_rec"
print(dir_path)
for subdir, dirs, files in os.walk(dir_path):
    print(subdir)
    print(dirs)
    print(files)
    dir_path = subdir + "/" + "IMRPhenomD_result.h5"
    print(dir_path)
    result = tupak.core.result.read_in_result(dir_path)

    with open('distance_evidence.dat', 'a') as outfile:
        outfile.write(str(subdir) + '\t' +
                      str(result.bayes_factor) + '\t' +
                      str(result.bayes_factor_err) + '\t' +
                      str(result.log_evidence) + '\t' +
                      str(result.log_evidence_err) + '\n')