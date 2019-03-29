import pandas as pd
import numpy as np
import sys

id = sys.argv[1]

outdir = id + '_reweighing_result/'
outfiles = ['0_7.json']

sampling_errors = np.array([])
reweighing_to_memory_errors = np.array([])
reweighing_from_memory_errors = np.array([])
combined_reweighing_errors = np.array([])

sampling_bfs = np.array([])
injection_bfs = np.array([])
reweighing_from_memory_bfs = np.array([])
reweighing_to_memory_bfs = np.array([])

for outfile in outfiles:
    table = pd.read_json(outdir + outfile)
    print(outfile)
    sampling_bfs = np.append(sampling_bfs, table.sampling_bfs)
    injection_bfs = np.append(injection_bfs, table.injection_bfs)
    reweighing_from_memory_bfs = np.append(reweighing_from_memory_bfs, table.reweighing_from_memory_bfs)
    reweighing_to_memory_bfs = np.append(reweighing_to_memory_bfs, table.reweighing_to_memory_bfs)

sampling_errors = np.append(sampling_errors, np.abs(injection_bfs - sampling_bfs))
reweighing_from_memory_errors = np.append(reweighing_from_memory_errors,
                                          np.abs(injection_bfs - reweighing_from_memory_bfs))
reweighing_to_memory_errors = np.append(reweighing_to_memory_errors,
                                        np.abs(injection_bfs - reweighing_to_memory_bfs))
combined_reweighing_errors = np.append(combined_reweighing_errors,
                                       np.abs(injection_bfs - 0.5 *
                                              (reweighing_to_memory_bfs + reweighing_from_memory_bfs)))

print(sampling_errors)
print(sampling_bfs)
print(injection_bfs)
print('Mean sampling log BF: \t' + str(np.mean(sampling_bfs[~np.isnan(sampling_bfs)])))
print('Mean injection log BF: \t' + str(np.mean(injection_bfs[~np.isnan(injection_bfs)])))
print('Mean reweighing from memory log BF: \t' + str(np.mean(reweighing_from_memory_bfs[~np.isnan(reweighing_from_memory_bfs)])))
print('Mean reweighing to memory log BF: \t' + str(np.mean(reweighing_to_memory_bfs[~np.isnan(reweighing_to_memory_bfs)])))
print('Mean combined log BF: \t' + str(np.mean(0.5*(reweighing_from_memory_bfs[~np.isnan(reweighing_from_memory_bfs)]
                                               + reweighing_to_memory_bfs[~np.isnan(reweighing_to_memory_bfs)]))))


print('Mean sampling error: \t' + str(np.mean(sampling_errors[~np.isnan(sampling_errors)])))
print('Mean reweighing from memory error: \t' + str(np.mean(reweighing_from_memory_errors[~np.isnan(reweighing_from_memory_errors)])))
print('Mean reweighing to memory error: \t' + str(np.mean(reweighing_to_memory_errors[~np.isnan(reweighing_to_memory_errors)])))
print('Mean combined reweighing error: \t' + str(np.mean(combined_reweighing_errors[~np.isnan(combined_reweighing_errors)])))

print('Total evidence sampling: \t' + str(np.sum(sampling_bfs[~np.isnan(sampling_bfs)])))
print('Total evidence injection: \t' + str(np.sum(injection_bfs[~np.isnan(sampling_bfs)])))
print('Total evidence reweighing to memory: \t' + str(np.sum(reweighing_to_memory_bfs[~np.isnan(sampling_bfs)])))
print('Total evidence reweighing from memory: \t' + str(np.sum(reweighing_from_memory_bfs[~np.isnan(sampling_bfs)])))