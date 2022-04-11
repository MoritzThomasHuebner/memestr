import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.kde import gaussian_kde

from memestr.events import events, precessing_events

precessing = False

if precessing:
    event_list = precessing_events
else:
    event_list = events

amps = np.linspace(-50, 50, 100)
amps_bin_edges = np.linspace(-50, 50, 101)
ln_probs = np.zeros(len(amps))
dx = amps[1] - amps[0]
unlogged_probs = np.ones(len(amps))

for event in event_list:
    event_name = event.name
    print(event_name)
    if precessing:
        event_name += "_2000"

    filename = f"memory_amplitude_results/{event_name}_memory_amplitude_posterior_50.txt"
    try:
        samples = np.array(pd.read_csv(f'memory_amplitude_results/{event_name}_memory_amplitude_posterior_50.csv')['amplitude_samples'])
    except OSError as e:
        print(e)
        continue
    probs, _ = np.histogram(samples, bins=amps_bin_edges, density=True)
    ln_probs += np.log(probs)
    unlogged_probs *= probs
    # plt.step(amps, probs)
    # plt.title(event)
    # plt.savefig(f"memory_amplitude_results/{event_name}_rebinned_posterior.png")
    # plt.clf()

probs = np.exp(ln_probs)
integrand = probs * dx
integrated_probs = np.sum(probs * dx)
probs /= integrated_probs
cdf = np.cumsum(probs)

print(amps[np.argmax(probs)])
print(amps[np.where(cdf < 0.05)[0][-1]])
print(amps[np.where(cdf > 0.95)[0][0]])

plt.step(amps, probs, where='mid')
plt.xlabel('memory amplitude')
plt.ylabel('probability')
plt.xlim(-50, 50)
plt.savefig(f'memory_amplitude_results/combined_result_alt_50.png')
plt.clf()


#
# plt.step(amps_bins, cdf, where='mid')
# plt.xlabel('memory amplitude')
# plt.ylabel('CDF')
# plt.savefig(f'memory_amplitude_results/combined_result_cdf_alt_50.png')
# plt.clf()


