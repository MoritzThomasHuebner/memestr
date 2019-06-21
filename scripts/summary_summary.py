import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import memestr
import numpy as np

minimums = np.arange(0, 1950, 50)
maximums = minimums + 50

memory_log_bfs = np.array([])
memory_log_bfs_injected = np.array([])
memory_log_bfs_injected_degenerate = np.array([])
hom_log_bfs = np.array([])
hom_log_bfs_injected = np.array([])
gw_log_bfs = np.array([])
gw_log_bfs_injected = np.array([])
for min_event_id, max_event_id in zip(minimums, maximums):
    memory_log_bfs = np.append(memory_log_bfs, np.loadtxt('summary_memory_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    memory_log_bfs_injected = np.append(memory_log_bfs_injected, np.loadtxt('summary_memory_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    memory_log_bfs_injected_degenerate = np.append(memory_log_bfs_injected_degenerate, np.loadtxt('summary_memory_log_bfs_injected_degenerate' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    hom_log_bfs = np.append(hom_log_bfs, np.loadtxt('summary_hom_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    hom_log_bfs_injected = np.append(hom_log_bfs_injected, np.loadtxt('summary_hom_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    gw_log_bfs = np.append(gw_log_bfs, np.loadtxt('summary_gw_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    gw_log_bfs_injected = np.append(gw_log_bfs_injected, np.loadtxt('summary_gw_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))

np.random.seed(42)
np.random.shuffle(memory_log_bfs_injected)
memory_log_bfs_injected_cumsum = np.cumsum(memory_log_bfs_injected)
np.random.seed(42)
np.random.shuffle(memory_log_bfs_injected_degenerate)
memory_log_bfs_injected_degenerate_cumsum = np.cumsum(memory_log_bfs_injected_degenerate)
np.random.seed(42)
np.random.shuffle(memory_log_bfs)
memory_log_bfs_cumsum = np.cumsum(memory_log_bfs)
np.random.seed(42)
np.random.shuffle(hom_log_bfs_injected)
hom_log_bfs_injected_cumsum = np.cumsum(hom_log_bfs_injected)
np.random.seed(42)
np.random.shuffle(hom_log_bfs)
hom_log_bfs_cumsum = np.cumsum(hom_log_bfs)
np.random.seed(42)
np.random.shuffle(gw_log_bfs_injected)
gw_log_bfs_injected_cumsum = np.cumsum(gw_log_bfs_injected)
np.random.seed(42)
np.random.shuffle(gw_log_bfs)
gw_log_bfs_cumsum = np.cumsum(gw_log_bfs)

plt.hist(hom_log_bfs, bins=45, label='Reweighted')
plt.hist(hom_log_bfs_injected, bins=45, label='Injected')
plt.xlabel('log BFs')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.savefig('summary_plot_hom_hist')
plt.clf()

plt.hist(memory_log_bfs, bins=45, label='Reweighted')
plt.hist(memory_log_bfs_injected, bins=45, label='Injected')
plt.xlabel('log BFs')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.savefig('summary_plot_memory_hist')
plt.clf()

plt.hist(gw_log_bfs, bins=45, label='Reweighted')
plt.hist(gw_log_bfs_injected, bins=45, label='Injected')
plt.xlabel('log BFs')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.savefig('summary_plot_gw_hist')
plt.clf()

plt.plot(memory_log_bfs_injected_cumsum, label='injected', linestyle='--')
# plt.plot(memory_log_bfs_injected_degenerate_cumsum, label='injected degenerate', linestyle='--')
plt.plot(memory_log_bfs_cumsum, label='sampled')
plt.xlabel('Event ID')
plt.ylabel('Cumulative log BF')
plt.ylim(-5, 12)
plt.legend()
plt.tight_layout()
plt.savefig('summary_plot_cumulative_memory_log_bf')
plt.clf()

plt.plot(hom_log_bfs_injected_cumsum, label='injected', linestyle='--')
plt.plot(hom_log_bfs_cumsum, label='sampled')
plt.xlabel('Event ID')
plt.ylabel('Cumulative log BF')
plt.legend()
plt.tight_layout()
plt.savefig('summary_plot_cumulative_hom_log_bf')
plt.clf()

plt.plot(gw_log_bfs_injected_cumsum, label='injected', linestyle='--')
plt.plot(gw_log_bfs_cumsum, label='sampled')
plt.xlabel('Event ID')
plt.ylabel('Cumulative log BF')
plt.legend()
plt.tight_layout()
plt.savefig('summary_plot_cumulative_gw_log_bf')
plt.clf()

n_effs = []
n_eff_fracs = []
for i in range(2000):
    try:
        pp_res = memestr.core.postprocessing.PostprocessingResult.from_json(str(i) + '_dynesty_production_IMR_non_mem_rec/')
        n_eff_frac = pp_res.effective_samples / len(pp_res.hom_weights)
        if np.isnan(n_eff_frac):
            n_eff_fracs.append(0)
            n_effs.append(1.)
        elif np.isinf(n_eff_frac):
            n_eff_fracs.append(0)
            n_effs.append(1.)
        else:
            n_eff_fracs.append(pp_res.effective_samples/len(pp_res.hom_weights))
            n_effs.append(pp_res.effective_samples)
        if n_effs[-1] < 4:
            print(i)
    except (AttributeError, FileNotFoundError):
        continue

plt.hist(n_eff_fracs, bins=45)
plt.xlabel('Fraction of effective samples')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('summary_plot_n_eff_frac_hist')
plt.clf()

plt.hist(n_effs, bins=45)
plt.xlabel('Number of effective samples')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('summary_plot_n_eff_hist')
plt.clf()

plt.plot(n_eff_fracs)
# plt.semilogy()
plt.xlabel('Event ID')
plt.ylabel('Effective sample fraction')
plt.tight_layout()
plt.savefig('summary_plot_n_eff_frac_vs_event_id')
plt.clf()

plt.plot(n_effs)
plt.semilogy()
plt.xlabel('Event ID')
plt.ylabel('Number of effective samples')
plt.tight_layout()
plt.savefig('summary_plot_n_eff_vs_event_id')
plt.clf()

# np.savetxt("n_effs", n_effs)
# n_effs_additional_runs = [int(80/x) for x in n_effs]
# with open("n_effs_additional_runs", 'w') as f:
#     for i in range(len(n_effs_additional_runs)):
#         print(i)
#         f.write(str(n_effs_additional_runs[i]) + '\n')
# print(sum(n_effs_additional_runs))
# np.savetxt("n_effs_additional_runs", n_effs_additional_runs)

