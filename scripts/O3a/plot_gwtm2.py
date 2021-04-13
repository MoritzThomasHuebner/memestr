import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.special import logsumexp
from collections import namedtuple
from memestr.events import events

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use('paper.mplstyle')

Event = namedtuple("Event", ["time_tag", "name", "detectors"])

outdir = "."
log_bfs = []
log_bfs_prec = []
log_bfs_prec_trimmed = []
plot_event_list = []
plot_prec_event_list = []

excluded_prec_events = ["GW151012", "GW190412", "GW190814", "GW190513A", "GW190707A",
                        "GW190728A", "GW190728B", "GW190924A", "GW190929A"]

for event in events:
    plot_event_list.append(event.name)
    try:
        log_memory_weights = np.loadtxt(f"{outdir}/{event.name}_memory_log_weights")

        reweighted_memory_log_bf = logsumexp(log_memory_weights) - np.log(len(log_memory_weights))
        log_bfs.append(reweighted_memory_log_bf)
        # n_eff_hom = np.sum(np.exp(log_memory_weights)) ** 2 / np.sum(np.exp(log_memory_weights) ** 2)
        # print(event)
        print(f"{event.name}\t{log_bfs[-1]}")
    except Exception as e:
        print(e)
        log_bfs.append(np.nan)

    try:
        if event.name in excluded_prec_events:
            raise Exception
        log_memory_weights_prec = np.loadtxt(f"{outdir}/{event.name}_prec_2000_memory_log_weights")
        log_memory_weights_prec_trimmed = log_memory_weights_prec[np.where(log_memory_weights_prec > -10)]
        reweighted_memory_log_bf_prec = logsumexp(log_memory_weights_prec) - np.log(len(log_memory_weights_prec))
        reweighted_memory_log_bf_prec_trimmed = logsumexp(log_memory_weights_prec_trimmed) - np.log(len(log_memory_weights_prec_trimmed))
        log_bfs_prec.append(reweighted_memory_log_bf_prec)
        log_bfs_prec_trimmed.append(reweighted_memory_log_bf_prec_trimmed)
        print(f"{event.name}\t{log_bfs_prec[-1]}")
        print(f"{event.name}\t{log_bfs_prec_trimmed[-1]}")
    except Exception as e:
        print(e)
        log_bfs_prec.append(np.nan)
        log_bfs_prec_trimmed.append(np.nan)


print(np.sum(np.nan_to_num(log_bfs, nan=0)))
print(np.sum(np.nan_to_num(log_bfs_prec, nan=0)))
print(np.sum(np.nan_to_num(log_bfs_prec_trimmed, nan=0)))

log_bfs_nrhybsur_gwtc_1 = np.loadtxt('O1O2_original_results.txt')

plt.figure(figsize=(18, 6))
markersize = 8
plt.plot(log_bfs, label=r'$\ln \mathrm{BF}_{\mathrm{mem}}$ IMRPhenomXHM', marker='H', linestyle='None', color='black', markersize=markersize)
plt.plot(log_bfs_prec, label=r'$\ln \mathrm{BF}_{\mathrm{mem}}$ NRSur7dq4', marker='P', linestyle='None', color='orange', markersize=markersize)
plt.plot(log_bfs_nrhybsur_gwtc_1, label=r'$\ln \mathrm{BF}_{\mathrm{mem}}$ NRHybSur3dq8 (Huebner et al. 2020)', marker='D', linestyle='None', color='blue', markersize=markersize, alpha=0.5)
# plt.plot(log_bfs_prec_trimmed, label='Memory ln BF NRSur7dq4 Trimmed', marker='D', linestyle='None', color='blue', markersize=markersize)
plt.grid(False)
plt.axhline(0, color='grey', linestyle='--')
plt.xticks(np.arange(len(plot_event_list)), tuple(plot_event_list), rotation=60)
plt.ylabel('$\ln \, \mathrm{BF}_{\mathrm{mem}}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("gwtc-2_trimmed.pdf")
plt.clf()
