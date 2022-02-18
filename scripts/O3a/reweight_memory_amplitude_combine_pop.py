import bilby.core.result
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from memestr.events import events, precessing_events
from memestr.postprocessing import reweight_by_memory_amplitude

precessing = False

if precessing:
    event_list = precessing_events
else:
    event_list = events

amps = np.linspace(-50, 50, 10000)
ln_probs = np.ones(len(amps))
dx = amps[1] - amps[0]

posteriors = []

for event in event_list:
    event_name = event.name
    if precessing:
        event_name += "_2000"
    try:
        posteriors.append(pd.read_csv(f'memory_amplitude_results/{event_name}_memory_amplitude_posterior.csv'))
    except Exception as e:
        print(e)
        break

combined_log_l = []


def calculate_inner_sum(memory_amplitude, posterior):
    return logsumexp(reweight_by_memory_amplitude(
        memory_amplitude=memory_amplitude, d_inner_h_mem=posterior['d_inner_h_mem'],
        optimal_snr_squared_h_mem=posterior['optimal_snr_squared_h_mem'],
        h_osc_inner_h_mem=posterior['h_osc_inner_h_mem']))


for a in amps:
    print(a)
    log_l = np.sum([calculate_inner_sum(a, posterior) - np.log(len(posterior))] for posterior in posteriors)
    combined_log_l.append(log_l)

res = dict(memory_amplitudes=amps, log_l=combined_log_l)
pd.DataFrame.from_dict(res).to_csv('memory_amplitude_results/combined_memory_amplitude_posterior_samples.csv')
