import numpy as np
import bilby
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams.update({'mathtext.fontset': 'dejavuserif'})

logger = bilby.core.utils.logger

event_ids = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608',
             'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823']
gwtc2_event_ids = ['GW190408A', 'GW190412', 'GW190413A', 'GW190413B', 'GW190421A', 'GW190424A', 'GW190426A', 'GW190503A', 'GW190512A',
                   'GW190513A', 'GW190514A', 'GW190517A', 'GW190519A', 'GW190521.1', 'GW190521B', 'GW190527A', 'GW190602A', 'GW190620A',
                   'GW190630A', 'GW190701A', 'GW190706A', 'GW190707A', 'GW190708A', 'GW190719A', 'GW190720A', 'GW190727A', 'GW190728A',
                   'GW190731A', 'GW190803A', 'GW190814', 'GW190828A', 'GW190828B', 'GW190909A', 'GW190910A', 'GW190915A', 'GW190924A',
                   'GW190929A', 'GW190930A']
event_ids.extend(gwtc2_event_ids)
memory_log_bfs = np.loadtxt('gwtc-1/real_event_log_bfs.txt')
memory_log_bfs = np.append(memory_log_bfs, np.zeros(len(gwtc2_event_ids)))
memory_log_bfs = np.array(memory_log_bfs)
cumulative_memory_log_bfs = np.cumsum(memory_log_bfs)
print(cumulative_memory_log_bfs[-1])
logger.info("Cumulative ln BF: " + str(cumulative_memory_log_bfs[-1]))
plt.figure(figsize=(16, 6))
plt.plot(memory_log_bfs, label='Memory ln BF', marker='H', linestyle='None', color='black')
plt.grid(False)
plt.axhline(0, color='grey', linestyle='--')
print(event_ids)
plt.xticks(np.arange(len(event_ids)), tuple(event_ids), rotation=75)
plt.ylabel('$\ln \, \mathcal{BF}_{\mathrm{mem}}$')
plt.tight_layout()
# plt.savefig('gwtc-1/gwtc-1.pdf')
plt.savefig('gwtc-2-test.pdf')
plt.clf()


# plt.plot(cumulative_memory_log_bfs, label='Cumulative memory log BFs', marker='H', color='black')
# plt.axhline(0, color='grey', linestyle='--')
# plt.xticks(np.arange(10), tuple(event_ids), rotation=70)
# plt.ylabel('Cumulative ln BF')
# plt.grid(False)
# plt.legend()
# plt.tight_layout()
# plt.savefig('gwtc-1/gwtc-1-cumulative.png')
# plt.clf()
#
