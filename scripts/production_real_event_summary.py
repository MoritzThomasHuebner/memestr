import numpy as np
import sys
import bilby
from scipy.misc import logsumexp

# event_id = sys.argv[1]
# number_of_parallel_runs = int(sys.argv[2])

event_id = 'GW150914'
number_of_parallel_runs = 32

base_result = bilby.result.read_in_result(filename=event_id + '/22_pe_result.json')
hom_like = np.array([])
memory_like = np.array([])
hom_weights = []


for run_id in range(number_of_parallel_runs):
    hom_like = np.append(hom_like, np.loadtxt(event_id + '/moritz_hom_log_likelihoods_' + str(run_id) + '.txt'))
    memory_like = np.append(hom_like, np.loadtxt(event_id + '/moritz_memory_log_likelihoods_' + str(run_id) + '.txt'))

for i in range(len(base_result.posterior)):
    hom_weights.append(hom_like[i] - base_result.posterior.log_likelihood.iloc[i])

memory_weights = memory_like - hom_like

hom_log_bf = logsumexp(hom_weights) - np.log(len(hom_weights))
memory_log_bf = logsumexp(memory_weights) - np.log(len(memory_weights))

print("HOM LOG BF: " + str(hom_log_bf))
print("Memory LOG BF: " + str(memory_log_bf))