import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from memestr.core.events import events


outdir = "."
log_bfs = []
plot_event_list = []

for event in events:
    try:
        log_memory_weights = np.loadtxt(f"{outdir}/{event.name}_memory_log_weights")
        reweighted_memory_log_bf = logsumexp(log_memory_weights) - np.log(len(log_memory_weights))
        n_eff_hom = np.sum(np.exp(log_memory_weights)) ** 2 / np.sum(np.exp(log_memory_weights) ** 2)
        log_bfs.append(reweighted_memory_log_bf)
        plot_event_list.append(event.name)
    except Exception as e:
        print(e)

print(np.sum(log_bfs))


plt.figure(figsize=(18, 6))
plt.plot(log_bfs, label='Memory ln BF', marker='H', linestyle='None', color='black')
plt.grid(False)
plt.axhline(0, color='grey', linestyle='--')
plt.xticks(np.arange(len(plot_event_list)), tuple(plot_event_list), rotation=60)
plt.ylabel('$\ln \, \mathrm{BF}_{\mathrm{mem}}$')
plt.tight_layout()
plt.savefig("gwtc-2.png")
plt.clf()
