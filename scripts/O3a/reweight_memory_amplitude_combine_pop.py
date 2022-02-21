import numpy as np
import pandas as pd
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from memestr.events import events, precessing_events
from memestr.postprocessing import reweight_by_memory_amplitude

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


amps = np.linspace(-50, 50, 1000)
dx = amps[1] - amps[0]


combined_log_l = []


def calculate_inner_sum(memory_amplitude, posterior):
    res = logsumexp(reweight_by_memory_amplitude(
        memory_amplitude=memory_amplitude, d_inner_h_mem=posterior['d_inner_h_mem'],
        optimal_snr_squared_h_mem=posterior['optimal_snr_squared_h_mem'],
        h_osc_inner_h_mem=posterior['h_osc_inner_h_mem']))
    return res


def calculate_outer_sum(memory_amplitude, posteriors):
    outer_sum = 0
    for posterior in posteriors:
        outer_sum += calculate_inner_sum(memory_amplitude, posterior) - np.log(len(posterior))
    return outer_sum


for a in amps:
    log_l = calculate_outer_sum(memory_amplitude=a, posteriors=posteriors)
    combined_log_l.append(log_l)
    print(f"{a}:\t{log_l}")

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


