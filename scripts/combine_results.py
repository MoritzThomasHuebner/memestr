from bilby.core.result import ResultList, read_in_result


# for i in range(2000):
i = 1999
outdir = "{}_dynesty_production_IMR_non_mem_rec/".format(i)
filename = "time_and_phase_shifted_result.json"
base_result = read_in_result(outdir + filename)
res_list = ResultList([base_result])
for j in range(41):
    try:
        res = read_in_result(outdir + str(i) + filename)
        res_list.append(res)
    except OSError:
        break
new_res = res_list.combine()
new_res.label = "time_and_phase_shifted_combined"
new_res.save_to_file()