import bilby
from bilby.core.result import ResultList, read_in_result

logger = bilby.core.utils.logger

for i in range(1850, 2000):
    outdir = "{}_dynesty_nr_sur_production_IMR_non_mem_rec/".format(i)
    res_list = ResultList([])
    for j in range(200, 208):
        try:
            res = read_in_result(outdir + "{}IMR_mem_inj_non_mem_rec_result.json".format(j))
            res_list.append(res)
        except OSError as e:
            logger.info(e)
            continue
    new_res = res_list.combine()
    new_res.label = "combined_proper_prior"
    new_res.save_to_file()
    new_res.plot_corner()

