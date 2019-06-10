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
ethan_hom_log_bfs = [-0.21, 0.31, -0.05, -0.10, -0.33, 1.15, -0.09, 0.11, 0.37, -0.25]

memory_log_bfs = []

for ethan_hom_log_bf, event_id in zip(ethan_hom_log_bfs, event_ids):
    base_result = bilby.result.read_in_result(filename=event_id + '/22_pe_result.json')
    hom_like = np.array([])
    memory_like = np.array([])
    hom_weights = []
    hom_memory_weights = []
    logger.info(event_id)
    try:
        for run_id in range(number_of_parallel_runs):
            hom_like = np.append(hom_like, np.loadtxt(event_id + '/moritz_hom_log_likelihoods_self_shifted_' + str(run_id) + '.txt'))
            memory_like = np.append(memory_like, np.loadtxt(event_id + '/moritz_memory_log_likelihoods_self_shifted_' + str(run_id) + '.txt'))
    except ValueError as e:
        logger.warning(e)
        logger.warning("Run ID: " + str(run_id))
        continue

    for i in range(len(base_result.posterior)):
        hom_weights.append(hom_like[i] - base_result.posterior.log_likelihood.iloc[i])
        hom_memory_weights.append(memory_like[i] - base_result.posterior.log_likelihood.iloc[i])

    # memory_weights = memory_like - hom_like

    hom_log_bf = logsumexp(hom_weights) - np.log(len(hom_weights))
    hom_memory_log_bf = logsumexp(hom_memory_weights) - np.log(len(hom_memory_weights))
    memory_log_bf = hom_memory_log_bf - hom_log_bf
    logger.info("Ethan HOM LOG BF: " + str(ethan_hom_log_bf))
    logger.info("Moritz HOM LOG BF: " + str(hom_log_bf))
    logger.info("Memory LOG BF: " + str(memory_log_bf))
    memory_log_bfs.append(memory_log_bf)

memory_log_bfs = np.array(memory_log_bfs)
cummulative_memory_log_bfs = np.cumsum(memory_log_bfs)

plt.plot(memory_log_bfs, label='Memory log BFs')
plt.plot(cummulative_memory_log_bfs, label='Cumulative memory log BFs')
plt.xticks(np.arange(10), tuple(event_ids), rotation=70)
plt.legend()
plt.tight_layout()
plt.savefig('money_plot')
plt.clf()