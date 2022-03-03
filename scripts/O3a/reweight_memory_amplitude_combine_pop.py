import bilby.core.result
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


for x in range(len(event_list)):
    posteriors = []
    event_name = event_list[x].name
    try:
        posterior = pd.read_csv(f'memory_amplitude_results/{event_name}_memory_amplitude_posterior_50.csv')
    except Exception:
        continue
    # posterior['d_inner_h_mem'] = np.array([complex(a) for a in np.array(posterior['d_inner_h_mem'])])
    posteriors.append(posterior)

    # for event in [event_list[x]]:
    # # for event in event_list:
    #     print(event.name)
    #     event_name = event.name
    #     if precessing:
    #         event_name += "_2000"
    #     try:
    #         posterior = pd.read_csv(f'memory_amplitude_results/{event_name}_memory_amplitude_posterior_50.csv')
    #         posterior['d_inner_h_mem'] = np.array([complex(a) for a in np.array(posterior['d_inner_h_mem'])])
    #         posteriors.append(posterior)
    #     except Exception as e:
    #         print(e)
    #         continue

    # def posterior_generator():
    #     for event in event_list[:1]:
    #         event_name = event.name
    #         if precessing:
    #             event_name += "_2000"
    #         try:
    #             posterior = pd.read_csv(f'memory_amplitude_results/{event_name}_memory_amplitude_posterior_50.csv')
    #             posterior['d_inner_h_mem'] = np.array([complex(a) for a in np.array(posterior['d_inner_h_mem'])])
    #             yield posterior
    #         except Exception as e:
    #             print(e)
    #             continue

    # amps = np.linspace(-50, 50, 1000)
    amps = np.linspace(-20, 20, 1000)
    dx = amps[1] - amps[0]

    result = bilby.core.result.read_in_result(
        f'{event_name}/result/run_data0_{event_list[x].time_tag}'
        f'_analysis_{event_list[x].detectors}_dynesty_merge_result.json')

    combined_log_l = reconstruct_memory_amplitude_population_posterior(
        memory_amplitudes=amps, posteriors=posteriors)
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

    plt.step(amps, probs, where='mid', label="reconstructed posterior Eq.11")
    amp_samples = np.array(posteriors[0]['amplitude_samples'])[np.where(np.abs(np.array(posteriors[0]['amplitude_samples'])) < 20 )[0]]
    plt.hist(amp_samples, bins='fd', density=True, alpha=0.5, label="Resampled posterior")
    amps_samples = np.array(result.posterior['memory_amplitude'])[np.where(np.abs(np.array(result.posterior['memory_amplitude'])) < 20 )[0]]
    plt.hist(amp_samples, bins='fd', density=True, alpha=0.5, label="Sampled posterior")
    # plt.hist(posteriors[0]['amplitude_samples'], bins='fd', density=True, alpha=0.5)
    plt.xlabel('memory amplitude')
    plt.ylabel('probability')
    plt.legend()
    # plt.savefig(f'memory_amplitude_results/combined_result.png')
    plt.savefig(f'memory_amplitude_results/{event_name}_memory_amplitude_posterior.png')
    plt.clf()

    # plt.step(amps, probs, where='mid')
    # plt.xlabel('memory amplitude')
    # plt.ylabel('probability')
    # plt.xlim(-50, 50)
    # plt.savefig(f'memory_amplitude_results/combined_result_zoomed.png')
    # plt.clf()
    #
    # plt.step(amps, cdf, where='mid')
    # plt.xlabel('memory amplitude')
    # plt.ylabel('CDF')
    # plt.savefig(f'memory_amplitude_results/combined_result_cdf.png')
    # plt.clf()


