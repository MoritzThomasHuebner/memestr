import numpy as np
import bilby
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams.update({'font.family': 'sans'})
logger = bilby.core.utils.logger

number_of_parallel_runs = 64
event_ids = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608',
             'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823']

memory_log_bfs = np.loadtxt('gwtc-1/real_event_log_bfs.txt')
memory_log_bfs = np.array(memory_log_bfs)
cumulative_memory_log_bfs = np.cumsum(memory_log_bfs)
logger.info("Cumulative ln BF: " + str(cumulative_memory_log_bfs[-1]))
plt.plot(memory_log_bfs, label='Memory ln BF', marker='H', linestyle='None', color='black')
plt.grid(False)
plt.axhline(0, color='grey', linestyle='--')
plt.xticks(np.arange(10), tuple(event_ids), rotation=45)
plt.ylabel('Memory ln BF')
plt.tight_layout()
plt.savefig('gwtc-1/gwtc-1-poster.pdf')
plt.clf()


plt.plot(cumulative_memory_log_bfs, label='Cumulative memory log BFs', marker='H', color='black')
plt.axhline(0, color='grey', linestyle='--')
plt.xticks(np.arange(10), tuple(event_ids), rotation=70)
plt.ylabel('Cumulative ln BF')
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('gwtc-1/gwtc-1-cumulative.png')
plt.clf()

