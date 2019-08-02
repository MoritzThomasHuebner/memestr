import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
label = sys.argv[1]

log_bfs = np.array([])
trials = np.array([])
snrs = np.array([])
memory_snrs = np.array([])

for i in range(256):
    try:
        data = np.loadtxt('Injection_log_bfs/Injection_log_bfs_{}_{}.txt'.format(label, i))
    except Exception:
        continue
    log_bfs = np.append(log_bfs, data[:, 0])
    trials = np.append(trials, data[:, 1])
    snrs = np.append(snrs, data[:, 2])
    memory_snrs = np.append(memory_snrs, data[:, 3])


print("Memory log BF per Event: " + str(-np.sum(log_bfs)/len(log_bfs)))
print("Events to log BF = 8: " + str(-8*len(log_bfs)/np.sum(log_bfs)))
print("Total number of events considered: " + str(len(log_bfs)))


sys.exit(0)
# for i in range(32, 97):
#     data = np.loadtxt('Injection_log_bfs/Injection_log_bfs_{}.txt'.format(i))
#     log_bfs = np.append(log_bfs, data[:, 0])
#     trials = np.append(trials, data[:, 1])

plt.scatter(log_bfs, memory_snrs)
plt.xlabel('log BFs')
plt.ylabel('Memory SNR')
plt.savefig('Injection_log_bfs/{}_log_bfs_vs_memory_snrs'.format(label))
# plt.show()
plt.clf()

plt.hist(log_bfs, bins=int(np.sqrt(len(log_bfs))))
plt.semilogy()
plt.xlabel('log BFs')
plt.savefig('Injection_log_bfs/{}_log_bfs_hist'.format(label))
# plt.show()
plt.clf()

plt.hist(snrs, bins=int(np.sqrt(len(snrs))))
plt.semilogy()
plt.xlabel('SNRs')
plt.savefig('Injection_log_bfs/{}_snrs_hist'.format(label))
# plt.show()
plt.clf()

plt.hist(trials, bins=int(np.sqrt(len(trials))))
plt.xlabel('trials')
plt.savefig('Injection_log_bfs/{}_trials_hist'.format(label))
# plt.show()
plt.clf()

# plt.plot(np.sort(snrs), np.linspace(1, 0, len(snrs)))
# plt.semilogy()
# plt.xlabel('SNR')
# plt.ylabel('fraction above SNR')
# plt.savefig('Injection_log_bfs/{}_snrs.format(label)')
# plt.show()
# plt.clf()

# import sys
# sys.exit(0)


maximums = []

for i in range(100000):
    print(i)
    reduced_log_bfs = np.random.choice(log_bfs, 100000)
    np.random.shuffle(reduced_log_bfs)
    reduced_log_bfs_cumsum = np.cumsum(reduced_log_bfs)
    maximums.append(np.max(reduced_log_bfs))
# minimums = np.loadtxt('minimums.txt')
np.savetxt('{}_maximums.txt'.format(label), maximums)
percentiles = np.percentile(maximums, [100 * (1 - 0.9999994), 100 * (1 - 0.99994), 0.3, 5, 32])
plt.hist(maximums, alpha=0.5, bins=100)
plt.xlim(15, -5)
colors = ['red', 'orange', 'cyan', 'black', 'green']
for i in range(len(percentiles)):
    plt.axvline(percentiles[i], label='${}\sigma$'.format(5 - i), color=colors[i])
plt.legend()
plt.xlabel('Minimum log BF')
plt.ylabel('Count')
plt.semilogy()
plt.savefig('Injection_log_bfs/{}_minimums'.format(label))
# plt.show()
plt.clf()
# maximums = []

# for i in range(1000000):
#     print(i)
#     np.random.shuffle(log_bfs)
#     maximums.append(np.max(np.cumsum(log_bfs)))
# maximums = np.loadtxt('maximums.txt')
# pcs = [99.99994, 99.994, 99.7, 95, 68]
# pcs = 100*(-1/np.linspace(1/.50, 1/.9999994, 100) + 1.4999994)
# percentiles = np.percentile(maximums, pcs)
# for percentage, log_bf in zip(pcs, percentiles):
#     print(str(1 - percentage/100) + '\t' + str(1/np.exp(log_bf)))
# plt.semilogy()
# plt.plot(1 - pcs/100, 1/np.exp(percentiles))
# plt.plot([0, 0.5], [0, 0.5])
# plt.show()
# plt.clf()
# plt.hist(maximums, alpha=0.5, bins=1000)
# plt.xlim(-5, 15)
# colors = ['red', 'orange', 'cyan', 'black', 'green']
# for i in range(len(percentiles)):
#     plt.axvline(percentiles[i], label='${}\sigma$'.format(5 - i), color=colors[i])
# plt.legend()
# plt.xlabel('Maximum log BF')
# plt.ylabel('Count')
# plt.semilogy()
# plt.show()
# plt.clf()

