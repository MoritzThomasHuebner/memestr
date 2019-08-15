import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import memestr
import scipy.stats
import numpy as np

minimums = np.arange(0, 2000, 50)
maximums = minimums + 50

memory_log_bfs = np.array([])
memory_log_bfs_injected = np.array([])
memory_log_bfs_injected_degenerate = np.array([])
hom_log_bfs = np.array([])
hom_log_bfs_injected = np.array([])
gw_log_bfs = np.array([])
gw_log_bfs_injected = np.array([])
for min_event_id, max_event_id in zip(minimums, maximums):
    memory_log_bfs = np.append(memory_log_bfs, np.loadtxt('summary/summary_memory_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    memory_log_bfs_injected = np.append(memory_log_bfs_injected, np.loadtxt('summary/summary_memory_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    invalid_indices = np.where(memory_log_bfs < -2)
    memory_log_bfs[invalid_indices] = 0
    memory_log_bfs_injected[invalid_indices] = 0
    # memory_log_bfs_injected_degenerate = np.append(memory_log_bfs_injected_degenerate, np.loadtxt('summary_memory_log_bfs_injected_degenerate' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    # hom_log_bfs = np.append(hom_log_bfs, np.loadtxt('summary_hom_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    # hom_log_bfs_injected = np.append(hom_log_bfs_injected, np.loadtxt('summary_hom_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    # gw_log_bfs = np.append(gw_log_bfs, np.loadtxt('summary_gw_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    # gw_log_bfs_injected = np.append(gw_log_bfs_injected, np.loadtxt('summary_gw_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))

np.random.seed(42)
np.random.shuffle(memory_log_bfs_injected)
memory_log_bfs_injected_cumsum = np.cumsum(memory_log_bfs_injected)
# np.random.seed(42)
# np.random.shuffle(memory_log_bfs_injected_degenerate)
# memory_log_bfs_injected_degenerate_cumsum = np.cumsum(memory_log_bfs_injected_degenerate)
np.random.seed(42)
np.random.shuffle(memory_log_bfs)
memory_log_bfs_cumsum = np.cumsum(memory_log_bfs)
# np.random.seed(42)
# np.random.shuffle(hom_log_bfs_injected)
# hom_log_bfs_injected_cumsum = np.cumsum(hom_log_bfs_injected)
# np.random.seed(42)
# np.random.shuffle(hom_log_bfs)
# hom_log_bfs_cumsum = np.cumsum(hom_log_bfs)
# np.random.seed(42)
# np.random.shuffle(gw_log_bfs_injected)
# gw_log_bfs_injected_cumsum = np.cumsum(gw_log_bfs_injected)
# np.random.seed(42)
# np.random.shuffle(gw_log_bfs)
# gw_log_bfs_cumsum = np.cumsum(gw_log_bfs)

# plt.hist(hom_log_bfs, bins=45, label='Reweighted')
# plt.hist(hom_log_bfs_injected, bins=45, label='Injected')
# plt.xlabel('log BFs')
# plt.ylabel('count')
# plt.legend()
# plt.tight_layout()
# plt.savefig('summary_plot_hom_hist_new')
# plt.clf()

# plt.hist(memory_log_bfs, bins=45, label='Reweighted')
# plt.hist(memory_log_bfs_injected, bins=45, label='Injected')
# plt.xlabel('log BFs')
# plt.ylabel('count')
# plt.legend()
# plt.tight_layout()
# plt.savefig('summary_plot_memory_hist_new')
# plt.clf()

# plt.hist(gw_log_bfs, bins=45, label='Reweighted')
# plt.hist(gw_log_bfs_injected, bins=45, label='Injected')
# plt.xlabel('log BFs')
# plt.ylabel('count')
# plt.legend()
# plt.tight_layout()
# plt.savefig('summary_plot_gw_hist_new')
# plt.clf()


label = 'snr_8_12_no_mem'

log_bfs = np.array([])
trials = np.array([])
snrs = np.array([])
memory_snrs = np.array([])

for i in range(256):
    try:
        data = np.loadtxt('Injection_log_bfs/Injection_log_bfs_{}_{}.txt'.format(label, i))
    except Exception:
        continue
    log_bfs = np.append(log_bfs, data[:, 0])
    trials = np.append(trials, data[:, 1])
    snrs = np.append(snrs, data[:, 2])
    memory_snrs = np.append(memory_snrs, data[:, 3])

# log_bfs[np.where(log_bfs > 8)] = 8
print("Memory log BF per Event: " + str(-np.sum(log_bfs)/len(log_bfs)))
print("Events to log BF = 8: " + str(-8*len(log_bfs)/np.sum(log_bfs)))
print("Total number of events considered: " + str(len(log_bfs)))

plt.hist(log_bfs, bins=int(np.sqrt(len(log_bfs))))
plt.semilogy()
plt.xlabel('Log BF')
plt.savefig('summary/summary_plot_log_bf_distribution.pdf')
plt.clf()


# log_bf_distribution = []
# for i in range(10000):
#     log_bf_distribution.append(-np.sum(np.random.choice(log_bfs, 2000)))
#
# plt.hist(log_bf_distribution, bins=100, normed=True)
# plt.axvline(np.mean(log_bf_distribution), color='red', label='Mean')
# plt.xlabel('Log BF after 2000 events')
# plt.ylabel('Probability (normalized)')
# plt.legend()
# plt.savefig('summary/summary_plot_log_bf')
# plt.clf()
#
# print(np.mean(log_bf_distribution))
# print(np.std(log_bf_distribution))

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'font.family': 'sans'})
plt.plot(memory_log_bfs_cumsum, label='This Memory Study')#, color='orange', linewidth=5.0)
arc = np.cumsum(-np.random.choice(log_bfs, 2000))
plt.plot(arc, alpha=0.3, color='grey', label='Other Realizations')
plt.axhline(8, linestyle='--', color='red', label='Detection Threshold')
for i in range(15):
    arc = np.cumsum(-np.random.choice(log_bfs, 2000))
    plt.plot(arc, alpha=0.3, color='grey')
# plt.plot(memory_log_bfs_injected_cumsum, label='injected', linestyle='--')
plt.xlabel('Event #')
plt.ylabel('Cumulative log BF')
plt.ylim(-7, 18)
plt.legend()
plt.tight_layout()
plt.savefig('summary/summary_plot_cumulative_memory_log_bf_poster.pdf')
plt.clf()
import sys
sys.exit(0)
required_events = []
for i in range(10000):
    tot = 0
    j = 0
    while tot < 8:
        tot -= np.sum(np.random.choice(log_bfs, 10))
        j += 10
    required_events.append(j)

interval = np.percentile(required_events, [5, 95])
plt.hist(required_events, bins=100, normed=True)
plt.axvline(np.mean(required_events), color='orange', label='Mean')
plt.axvline(interval[0], color='red', linestyle='--', label='$95\% \mathrm{CL}$')
plt.axvline(interval[1], color='red', linestyle='--')
plt.xlabel('Number of events to $\log \mathcal{BF} > 8$')
plt.ylabel('Probability (normalized)')
plt.legend()
plt.savefig('summary/summary_plot_required_events')
plt.clf()
print('Mean number of events: ' + str(np.mean(required_events)))
print('Median number of events: ' + str(np.median(required_events)))
print('Standard deviation on number of events: ' + str(np.std(required_events)))
print('95 percent CL on number of required events: ' + str(interval))


# plt.axvline(1850, label='Change of methods', linestyle=':', color='red')
# plt.ylim(-5, 12)

# plt.plot(memory_log_bfs_injected_cumsum, label='injected', linestyle='--')
# plt.plot(memory_log_bfs_injected_degenerate_cumsum, label='injected degenerate', linestyle='--')
# plt.plot(memory_log_bfs_cumsum, label='sampled')
# plt.axvline(1850, label='Change of methods', linestyle=':', color='red')
# plt.xlabel('Event ID')
# plt.ylabel('Cumulative log BF')
# plt.ylim(-5, 12)
# plt.xlim(1850, 2000)
# plt.legend()
# plt.tight_layout()
# plt.savefig('summary/summary_plot_cumulative_memory_log_bf_high_snr')
# plt.clf()


# plt.plot(hom_log_bfs_injected_cumsum, label='injected', linestyle='--')
# plt.plot(hom_log_bfs_cumsum, label='sampled')
# plt.xlabel('Event ID')
# plt.ylabel('Cumulative log BF')
# plt.legend()
# plt.tight_layout()
# plt.savefig('summary/summary_plot_cumulative_hom_log_bf')
# plt.clf()

# plt.plot(gw_log_bfs_injected_cumsum, label='injected', linestyle='--')
# plt.plot(gw_log_bfs_cumsum, label='sampled')
# plt.xlabel('Event ID')
# plt.ylabel('Cumulative log BF')
# plt.legend()
# plt.tight_layout()
# plt.savefig('summary/summary_plot_cumulative_gw_log_bf')
# plt.clf()

# n_effs = []
# n_eff_fracs = []
# for i in range(2000):
#     try:
#         pp_res = memestr.core.postprocessing.PostprocessingResult.from_json(str(i) + '_dynesty_production_IMR_non_mem_rec/')
#         n_eff_frac = pp_res.effective_samples / len(pp_res.hom_weights)
#         if np.isnan(n_eff_frac):
#             n_eff_fracs.append(0)
#             n_effs.append(1.)
#         elif np.isinf(n_eff_frac):
#             n_eff_fracs.append(0)
#             n_effs.append(1.)
#         else:
#             n_eff_fracs.append(pp_res.effective_samples/len(pp_res.hom_weights))
#             n_effs.append(pp_res.effective_samples)
#         if n_effs[-1] < 2:
#             print(i)
#     except (AttributeError, FileNotFoundError):
#         continue
#
# plt.hist(n_eff_fracs, bins=45)
# plt.xlabel('Fraction of effective samples')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.savefig('summary/summary_plot_n_eff_frac_hist')
# plt.clf()
#
# plt.hist(n_effs, bins=45)
# plt.xlabel('Number of effective samples')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.savefig('summary/summary_plot_n_eff_hist')
# plt.clf()
#
# plt.plot(n_eff_fracs)
# plt.semilogy()
# plt.xlabel('Event ID')
# plt.ylabel('Effective sample fraction')
# plt.tight_layout()
# plt.savefig('summary/summary_plot_n_eff_frac_vs_event_id')
# plt.clf()

# plt.plot(n_effs)
# plt.semilogy()
# plt.xlabel('Event ID')
# plt.ylabel('Number of effective samples')
# plt.tight_layout()
# plt.savefig('summary/summary_plot_n_eff_vs_event_id')
# plt.clf()

# np.savetxt("n_effs", n_effs)
# n_effs_additional_runs = [int(80/x) for x in n_effs]
# with open("n_effs_additional_runs", 'w') as f:
#     for i in range(len(n_effs_additional_runs)):
#         print(i)
#         f.write(str(n_effs_additional_runs[i]) + '\n')
# print(sum(n_effs_additional_runs))
# np.savetxt("n_effs_additional_runs", n_effs_additional_runs)

