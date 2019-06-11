import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

minimums = np.arange(0, 2000, 50)
maximums = minimums + 50
print(minimums)
print(maximums)
memory_log_bfs = np.array([])
memory_log_bfs_injected = np.array([])
hom_log_bfs = np.array([])
hom_log_bfs_injected = np.array([])
for min_event_id, max_event_id in zip(minimums, maximums):
    memory_log_bfs = np.append(memory_log_bfs, np.loadtxt('summary_memory_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    memory_log_bfs_injected = np.append(memory_log_bfs_injected, np.loadtxt('summary_memory_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    hom_log_bfs = np.append(hom_log_bfs, np.loadtxt('summary_hom_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))
    hom_log_bfs_injected = np.append(hom_log_bfs_injected, np.loadtxt('summary_hom_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt'))

memory_log_bfs_injected_cumsum = np.cumsum(memory_log_bfs_injected)
memory_log_bfs_cumsum = np.cumsum(memory_log_bfs)
hom_log_bfs_injected_cumsum = np.cumsum(hom_log_bfs_injected)
hom_log_bfs_cumsum = np.cumsum(hom_log_bfs)

plt.hist(hom_log_bfs, bins=45)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('HOM log BFs')
plt.savefig('summary_hom_hist')
plt.clf()

plt.hist(memory_log_bfs, bins=45)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('Memory log BFs')
plt.tight_layout()
plt.savefig('summary_memory_hist')
plt.clf()

plt.hist(memory_log_bfs_injected, bins=45)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('Memory log BFs injected')
plt.tight_layout()
plt.savefig('summary_memory_hist_injected')
plt.clf()

plt.plot(memory_log_bfs_injected_cumsum, label='injected', linestyle='--')
plt.plot(memory_log_bfs_cumsum, label='sampled')
plt.xlabel('Event ID')
plt.ylabel('Cumulative log BF')
plt.legend()
plt.tight_layout()
plt.savefig('summary_cumulative_memory_log_bf' + str(min_event_id) + '_' + str(max_event_id))
plt.clf()

plt.plot(hom_log_bfs_injected_cumsum, label='injected', linestyle='--')
plt.plot(hom_log_bfs_cumsum, label='sampled')
plt.xlabel('Event ID')
plt.ylabel('Cumulative log BF')
plt.legend()
plt.tight_layout()
plt.savefig('summary_cumulative_hom_log_bf' + str(min_event_id) + '_' + str(max_event_id))
plt.clf()

