import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from memestr.core.utils import _get_matched_filter_snrs

ids = ['032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047']
distances = [200, 230, 262, 299, 342, 391, 448, 512, 586, 670, 766, 876, 1002, 1147, 1311, 1500]

matched_filter_snrs = _get_matched_filter_snrs(distances)

average_injection_log_bfs = []

average_sampling_log_bfs = []
average_reweighing_to_memory_log_bfs = []
average_reweighing_from_memory_log_bfs = []

population_sampling_log_bfs_uncertainties = []
population_reweighing_to_memory_log_bfs_uncertainties = []
population_reweighing_from_memory_log_bfs_uncertainties = []


for idx, distance in zip(ids, distances):
    outdir = idx + '_reweighing_result/'
    outfiles = ['0_7.json']

    sampling_errors = np.array([])
    reweighing_to_memory_errors = np.array([])
    reweighing_from_memory_errors = np.array([])
    combined_reweighing_errors = np.array([])

    sampling_log_bfs = np.array([])
    injection_log_bfs = np.array([])
    reweighing_from_memory_log_bfs = np.array([])
    reweighing_to_memory_log_bfs = np.array([])

    for outfile in outfiles:
        table = pd.read_json(outdir + outfile)
        # print(outfile)
        sampling_log_bfs = np.append(sampling_log_bfs, table.sampling_bfs)
        injection_log_bfs = np.append(injection_log_bfs, table.injection_bfs)
        reweighing_from_memory_log_bfs = np.append(reweighing_from_memory_log_bfs,
                                                   table.reweighing_from_memory_bfs_mem_inj)
        reweighing_to_memory_log_bfs = np.append(reweighing_to_memory_log_bfs,
                                                 table.reweighing_to_memory_bfs_mem_inj)

    sampling_errors = np.append(sampling_errors, np.abs(injection_log_bfs - sampling_log_bfs))
    reweighing_from_memory_errors = np.append(reweighing_from_memory_errors,
                                              np.abs(injection_log_bfs - reweighing_from_memory_log_bfs))
    reweighing_to_memory_errors = np.append(reweighing_to_memory_errors,
                                            np.abs(injection_log_bfs - reweighing_to_memory_log_bfs))
    combined_reweighing_errors = np.append(combined_reweighing_errors,
                                           np.abs(injection_log_bfs - 0.5 *
                                                  (reweighing_to_memory_log_bfs + reweighing_from_memory_log_bfs)))

    print('Luminosity distance: ' + str(distance))
    print('Mean sampling log BF: \t' + str(np.mean(sampling_log_bfs[~np.isnan(sampling_log_bfs)])))
    print('Mean injection log BF: \t' + str(np.mean(injection_log_bfs[~np.isnan(injection_log_bfs)])))
    print('Mean reweighing from memory log BF: \t' + str(
        np.mean(reweighing_from_memory_log_bfs[~np.isnan(reweighing_from_memory_log_bfs)])))
    print('Mean reweighing to memory log BF: \t' + str(
        np.mean(reweighing_to_memory_log_bfs[~np.isnan(reweighing_to_memory_log_bfs)])))
    print('Mean combined log BF: \t' +
          str(0.5 * (np.mean(reweighing_from_memory_log_bfs[~np.isnan(reweighing_from_memory_log_bfs)]) +
                     np.mean(reweighing_to_memory_log_bfs[~np.isnan(reweighing_to_memory_log_bfs)]))))

    print('Mean sampling error: \t' + str(np.mean(sampling_errors[~np.isnan(sampling_errors)])))
    print('Mean reweighing from memory error: \t' + str(
        np.mean(reweighing_from_memory_errors[~np.isnan(reweighing_from_memory_errors)])))
    print('Mean reweighing to memory error: \t' + str(
        np.mean(reweighing_to_memory_errors[~np.isnan(reweighing_to_memory_errors)])))
    print('Mean combined reweighing error: \t' + str(
        np.mean(combined_reweighing_errors[~np.isnan(combined_reweighing_errors)])))
    print('\n')

    average_injection_log_bfs.append(injection_log_bfs[0])

    average_sampling_log_bfs.append(np.mean(sampling_log_bfs))
    average_reweighing_to_memory_log_bfs.append(np.mean(reweighing_to_memory_log_bfs))
    average_reweighing_from_memory_log_bfs.append(np.mean(reweighing_from_memory_log_bfs))

    population_sampling_log_bfs_uncertainties.append(np.mean(np.std(sampling_log_bfs)) / np.sqrt(len(sampling_log_bfs)))
    population_reweighing_to_memory_log_bfs_uncertainties.append(np.mean(np.std(reweighing_to_memory_log_bfs)) / np.sqrt(len(reweighing_to_memory_log_bfs)))
    population_reweighing_from_memory_log_bfs_uncertainties.append(np.mean(np.std(reweighing_from_memory_log_bfs)) / np.sqrt(len(reweighing_from_memory_log_bfs)))


print('\n')
print(average_injection_log_bfs)
print('\n')
print(average_sampling_log_bfs)
print(population_sampling_log_bfs_uncertainties)
print('\n')
print(average_reweighing_to_memory_log_bfs)
print(population_reweighing_to_memory_log_bfs_uncertainties)
print('\n')
print(average_reweighing_from_memory_log_bfs)
print(population_reweighing_from_memory_log_bfs_uncertainties)

plt.plot(matched_filter_snrs, average_injection_log_bfs, label='Log L at injected value')
plt.errorbar(x=matched_filter_snrs, y=average_sampling_log_bfs, fmt='v',
             yerr=np.array(population_sampling_log_bfs_uncertainties), label='Sampling')
plt.plot(x=matched_filter_snrs, y=average_reweighing_to_memory_log_bfs, fmt='s',
         label='Reweighing osc recovered')
plt.plot(x=matched_filter_snrs, y=average_reweighing_from_memory_log_bfs, fmt='o',
         label='Reweighing osc+memory\n recovered')
plt.legend()
plt.tight_layout()
plt.xlabel('Matched filter SNR')
plt.ylabel('log BF')
plt.savefig('LogBFs vs SNR')
plt.show()
plt.clf()

plt.plot(matched_filter_snrs, population_sampling_log_bfs_uncertainties, 'v',
         label='Sampling uncertainty')
plt.plot(matched_filter_snrs, population_reweighing_to_memory_log_bfs_uncertainties, 's',
         label='Reweighing osc stat. uncertainty')
plt.plot(matched_filter_snrs, population_reweighing_from_memory_log_bfs_uncertainties, 'o',
         label='Reweighing osc+memory \nstat. uncertainty')
plt.semilogy()
plt.tight_layout()
plt.legend()
plt.xlabel('Matched filter SNR')
plt.ylabel('$\Delta$ log BF')
plt.savefig('LogBFs errors vs SNR')
plt.show()
plt.clf()
