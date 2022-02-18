import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.kde import gaussian_kde

from memestr.events import events, precessing_events

precessing = False

if precessing:
    event_list = precessing_events
else:
    event_list = events

amps = np.linspace(-500, 500, 1000)
ln_probs = np.ones(len(amps))
dx = amps[1] - amps[0]

for event in event_list:
    event_name = event.name
    print(event_name)
    if precessing:
        event_name += "_2000"

    filename = f"memory_amplitude_results/{event_name}_memory_amplitude_posterior.txt"
    try:
        samples = np.loadtxt(f"memory_amplitude_results/{event_name}_memory_amplitude_posterior.txt")
    except OSError as e:
        print(e)
        break
    probs, edges = np.histogram(samples, bins=1000, density=True)
    ln_probs += np.log(probs)

    plt.step(amps, probs)
    plt.title(event)
    plt.savefig(f"memory_amplitude_results/{event_name}_rebinned_posterior.png")
    plt.clf()

probs = np.exp(ln_probs)
integrand = probs * dx
integrated_probs = np.sum(probs * dx)
probs /= integrated_probs
cdf = np.cumsum(probs)

plt.plot(amps, probs)
plt.xlabel('memory amplitude')
plt.ylabel('probability')
plt.savefig(f'memory_amplitude_results/combined_result.png')
plt.clf()

plt.plot(amps, probs)
plt.xlabel('memory amplitude')
plt.ylabel('probability')
plt.xlim(-50, 50)
plt.savefig(f'memory_amplitude_results/combined_result_zoomed.png')
plt.clf()

plt.plot(amps, cdf)
plt.xlabel('memory amplitude')
plt.ylabel('CDF')
plt.savefig(f'memory_amplitude_results/combined_result_cdf.png')
plt.clf()


