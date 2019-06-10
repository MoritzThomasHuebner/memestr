import numpy as np
import sys
import bilby
from scipy.misc import logsumexp
import matplotlib.pyplot as plt

logger = bilby.core.utils.logger


# event_id = sys.argv[1]
# number_of_parallel_runs = int(sys.argv[2])

# event_id = 'GW170823'
number_of_parallel_runs = 64
event_ids = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608',
             'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823']
# ethan_hom_log_bfs = [-0.21, 0.31, -0.05, -0.10, -0.33, 1.15, -0.09, 0.11, 0.37, -0.25]

memory_log_bfs = []

for event_id in event_ids:
    ethan_result = np.loadtxt(event_id + '/new_likelihoods.dat')
    ethan_22_log_likelihood = ethan_result[:, 0]
    ethan_posterior_hom_log_likelihood = ethan_result[:, 1]
    ethan_weight_log_likelihood = ethan_result[:, 2]

    base_result = bilby.result.read_in_result(filename=event_id + '/22_pe_result.json')
    hom_like_moritz = np.array([])
    hom_like_ethan = np.array([])
    memory_like = np.array([])
    hom_weights_moritz = []
    hom_weights_ethan = []
    hom_memory_weights = []
    logger.info(event_id)
    try:
        for run_id in range(number_of_parallel_runs):
            hom_like_moritz = np.append(hom_like_moritz, np.loadtxt(event_id + '/moritz_hom_log_likelihoods_' + str(run_id) + '.txt'))
            hom_like_ethan = np.append(hom_like_ethan, np.loadtxt(event_id + '/ethan_hom_log_likelihoods_' + str(run_id) + '.txt'))
            memory_like = np.append(memory_like, np.loadtxt(event_id + '/moritz_memory_log_likelihoods_' + str(run_id) + '.txt'))
    except ValueError as e:
        logger.warning(e)
        logger.warning("Run ID: " + str(run_id))
        continue

    for i in range(len(base_result.posterior)):
        hom_weights_moritz.append(hom_like_moritz[i] - base_result.posterior.log_likelihood.iloc[i])
        hom_weights_ethan.append(hom_like_ethan[i] - base_result.posterior.log_likelihood.iloc[i])
        hom_memory_weights.append(memory_like[i] - base_result.posterior.log_likelihood.iloc[i])

    # memory_weights = memory_like - hom_like

    hom_log_bf_moritz = logsumexp(hom_weights_moritz) - np.log(len(hom_weights_moritz))
    hom_log_bf_ethan = logsumexp(hom_weights_ethan) - np.log(len(hom_weights_ethan))
    hom_log_bf_ethan_posterior = np.log(np.sum(ethan_weight_log_likelihood)) - np.log(len(ethan_weight_log_likelihood))
    hom_memory_log_bf = logsumexp(hom_memory_weights) - np.log(len(hom_memory_weights))
    memory_log_bf = hom_memory_log_bf - hom_log_bf_moritz
    logger.info("Ethan Posterior HOM LOG BF: " + str(hom_log_bf_ethan_posterior))
    logger.info("Ethan Restored HOM LOG BF: " + str(hom_log_bf_ethan))
    logger.info("Moritz HOM LOG BF: " + str(hom_log_bf_moritz))
    logger.info("Memory LOG BF: " + str(memory_log_bf))
    memory_log_bfs.append(memory_log_bf)

memory_log_bfs = np.array(memory_log_bfs)
cumulative_memory_log_bfs = np.cumsum(memory_log_bfs)

plt.plot(memory_log_bfs, label='Memory log BFs')
plt.plot(cumulative_memory_log_bfs, label='Cumulative memory log BFs')
plt.xticks(np.arange(10), tuple(event_ids), rotation=70)
plt.legend()
plt.tight_layout()
plt.savefig('gwtm-1')
plt.clf()
