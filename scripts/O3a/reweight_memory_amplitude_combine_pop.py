import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memestr.events import events, precessing_events
from memestr.postprocessing import reconstruct_memory_amplitude_population_posterior

precessing = False

if precessing:
    event_list = precessing_events
else:
    event_list = events

posteriors = []

for event in event_list:
    event_name = event.name
    if precessing:
        event_name += "_2000"
    try:
        posterior = pd.read_csv(f'memory_amplitude_results/{event_name}_memory_amplitude_posterior.csv')
        posterior['d_inner_h_mem'] = np.array([complex(a) for a in np.array(posterior['d_inner_h_mem'])])
        posteriors.append(posterior)
    except Exception as e:
        print(e)
        continue
    break


amps = np.linspace(-500, 500, 1000)
dx = amps[1] - amps[0]


combined_log_l = reconstruct_memory_amplitude_population_posterior(memory_amplitudes=amps, posteriors=posteriors)
res = dict(memory_amplitudes=amps, log_l=combined_log_l)
pd.DataFrame.from_dict(res).to_csv('memory_amplitude_results/combined_memory_amplitude_posterior_pdf.csv', index=False)
df = pd.read_csv('memory_amplitude_results/combined_memory_amplitude_posterior_pdf.csv')
amps = np.array(df['memory_amplitudes'])
combined_log_l = np.array(df['log_l'])

combined_log_l -= np.max(combined_log_l)
probs = np.exp(combined_log_l)
probs = np.nan_to_num(probs)
integrand = probs * dx
integrated_probs = np.sum(probs * dx)
probs /= integrated_probs

cdf = np.cumsum(probs)
cdf /= cdf[-1]
print(amps[np.argmax(probs)])
print(amps[np.where(cdf < 0.05)[0][-1]])
print(amps[np.where(cdf > 0.95)[0][0]])

plt.step(amps, probs, where='mid')
plt.hist(posteriors[0]['amplitude_samples'], bins='fd', density=True, alpha=0.5)
plt.xlabel('memory amplitude')
plt.ylabel('probability')
plt.savefig(f'memory_amplitude_results/combined_result.png')
plt.clf()

plt.step(amps, probs, where='mid')
plt.xlabel('memory amplitude')
plt.ylabel('probability')
plt.xlim(-50, 50)
plt.savefig(f'memory_amplitude_results/combined_result_zoomed.png')
plt.clf()

plt.step(amps, cdf, where='mid')
plt.xlabel('memory amplitude')
plt.ylabel('CDF')
plt.savefig(f'memory_amplitude_results/combined_result_cdf.png')
plt.clf()


