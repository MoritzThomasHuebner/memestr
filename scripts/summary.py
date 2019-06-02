import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

memory_log_bfs = []
hom_log_bfs = []

for i in range(1000, 2000):
    try:
        memory_log_bf = np.loadtxt(str(i) + '_pypolychord_production_IMR_non_mem_rec/memory_log_bf.txt')
        hom_log_bf = np.loadtxt(str(i) + '_pypolychord_production_IMR_non_mem_rec/hom_log_bf.txt')
        memory_log_bfs.append(memory_log_bf)
        hom_log_bfs.append(hom_log_bf)
    except OSError as e:
        print(e)
        continue

memory_log_bfs = np.array(memory_log_bfs)
np.random.shuffle(memory_log_bfs)
memory_log_bfs_cumsum = np.cumsum(memory_log_bfs)

hom_log_bfs = np.array(hom_log_bfs)

plt.hist(hom_log_bfs, bins=30)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('HOM log BFs')
plt.savefig('summary_hom_hist')
plt.clf()

plt.hist(memory_log_bfs, bins=30)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('Memory log BFs')
plt.savefig('summary_memory_hist')
plt.clf()

plt.plot(memory_log_bfs_cumsum)
plt.xlabel('Event ID')
plt.ylabel('Cummulative log BF')
plt.savefig('summary_cummulative_memory_log_bf')
plt.clf()