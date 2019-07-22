import bilby
from bilby.core.result import ResultList, read_in_result

logger = bilby.core.utils.logger

# for i in range(1950, 2000):
#     outdir = "{}_dynesty_production_IMR_non_mem_rec/".format(i)
#     filename = "reconstructed_result_result.json"
#     base_result = read_in_result(outdir + filename)
#     res_list = ResultList([base_result])
#     for j in range(81):
#         try:
#             res = read_in_result(outdir + "reconstructed_result{}_result.json".format(j))
#             res_list.append(res)
#         except OSError as e:
#             logger.info(e)
#             continue
#     new_res = res_list.combine()
#     new_res.label = "reconstructed_combined"
#     new_res.save_to_file()
    # new_res.plot_corner()

for i in [1863, 1870, 1876, 1896, 1903, 1907, 1912, 1916, 1936, 1937, 1938, 1958, 1971, 1982, 1996]:
    outdir = "{}_dynesty_nrsur_rec_IMR_inj_production_IMR_non_mem_rec/".format(i)
    res_list = ResultList([])
    for j in range(100, 110):
        try:
            res = read_in_result(outdir + "{}IMR_mem_inj_non_mem_rec_result.json".format(j))
            res_list.append(res)
        except OSError as e:
            logger.info(e)
            continue
    new_res = res_list.combine()
    new_res.label = "combined_high_nlive"
    new_res.save_to_file()
    new_res.plot_corner()

