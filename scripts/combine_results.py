import bilby
from bilby.core.result import ResultList, read_in_result

logger = bilby.core.utils.logger

# for i in range(2000):
# print(i)
i = 1999
outdir = "{}_dynesty_production_IMR_non_mem_rec/".format(i)
filename = "reconstructed_result{}_result.json"
base_result = read_in_result(outdir + filename)
res_list = ResultList([base_result])
for j in range(41):
    try:
        res = read_in_result(outdir + filename.format(i))
        res_list.append(res)
    except OSError as e:
        logger.info(e)
        break
new_res = res_list.combine()
new_res.label = "reconstructed_combined"
new_res.save_to_file()
# new_res.plot_corner()