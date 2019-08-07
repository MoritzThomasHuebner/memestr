import numpy as np
import matplotlib
# matplotlib.use('Agg')
import scipy.stats
import matplotlib.pyplot as plt
import sys
# label = sys.argv[1]
# label = 'aplus'
label = 'snr_8_12_no_mem'

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

print("Memory log BF per Event: " + str(np.mean(log_bfs)))
print("Standard deviation log BF: " + str(np.std(log_bfs)))
print("Events to log BF = 8: " + str(8*len(log_bfs)/np.sum(log_bfs)))
print("Total number of events considered: " + str(len(log_bfs)))


plt.hist(log_bfs, bins='fd')
plt.semilogy()
plt.show()
plt.clf()


# idxs = np.where(snrs < 32)
# log_bfs[idxs] = 0
# assert False

# log_bf_distribution = []
# for i in range(10000):
#     log_bf_distribution.append(-np.sum(np.random.choice(log_bfs, 2000)))
#
# plt.hist(log_bf_distribution, bins=100)
# plt.show()
# plt.clf()
#
# print(np.mean(log_bf_distribution))
# print(np.std(log_bf_distribution))


# for i in range(20):
#     arc = np.cumsum(-np.random.choice(log_bfs, 2000))
#     plt.plot(arc, alpha=0.2, color='grey')
#     plt.axhline(8, linestyle='--', color='red')

# required_events = []
# for i in range(10000):
#     tot = 0
#     j = 0
#     while tot < 8:
#         tot -= np.sum(np.random.choice(log_bfs, 10))
#         j += 10
#     required_events.append(j)

# interval = scipy.stats.t.interval(0.95,
#                                   len(required_events)-1,
#                                   loc=np.mean(required_events),
#                                   scale=scipy.stats.sem(required_events))
#
# plt.hist(required_events, bins=100)
# plt.axvline(np.mean(required_events), color='orange')
# plt.axvline(interval[0], color='red', linestyle='--')
# plt.axvline(interval[1], color='red', linestyle='--')
# plt.clf()
# print('Mean number of events: ' + str(np.mean(required_events)))
# print('Median number of events: ' + str(np.median(required_events)))
# print('Standard deviation on number of events: ' + str(np.std(required_events)))

# sys.exit(0)
# for i in range(32, 97):
#     data = np.loadtxt('Injection_log_bfs/Injection_log_bfs_{}.txt'.format(i))
#     log_bfs = np.append(log_bfs, data[:, 0])
#     trials = np.append(trials, data[:, 1])

# plt.scatter(log_bfs, memory_snrs)
# plt.xlabel('log BFs')
# plt.ylabel('Memory SNR')
# plt.savefig('Injection_log_bfs/{}_log_bfs_vs_memory_snrs'.format(label))
# plt.clf()
#
plt.hist(snrs, bins='fd')
plt.semilogy()
plt.xlabel('SNRs')
plt.savefig('Injection_log_bfs/{}_snrs_hist'.format(label))
plt.clf()

plt.hist(trials, bins='fd')
plt.xlabel('trials')
plt.savefig('Injection_log_bfs/{}_trials_hist'.format(label))
plt.clf()

# assert False

print('Number of events to draw: ' + str(25*np.std(log_bfs)**2/np.mean(log_bfs)**2))

maximums = []
for i in range(100000):
    print(i)
    realization = -np.random.choice(log_bfs, 15000)
    realization_cumsum = np.cumsum(realization)
    maximums.append(np.max(realization_cumsum))
np.savetxt('Injection_log_bfs/{}_maximums.txt'.format(label), maximums)
# maximums = np.loadtxt('Injection_log_bfs/{}_maximums.txt'.format(label))

percentiles = np.percentile(maximums, [99.99994, 99.994, 99.7, 95, 68])
plt.hist(maximums, alpha=0.5, bins='fd')
colors = ['red', 'orange', 'cyan', 'black', 'green']
for i in range(len(percentiles)):
    plt.axvline(percentiles[i], label='${}\sigma$'.format(5 - i), color=colors[i])
plt.legend()
plt.xlabel('Maximum log BF')
plt.ylabel('Count')
plt.semilogy()
plt.savefig('Injection_log_bfs/{}_maximums'.format(label))
plt.clf()

for i in range(20):
    realization = np.random.choice(log_bfs, 15000)
    realization_cumsum = np.cumsum(realization)
    plt.plot(realization_cumsum, color='grey', alpha=0.3)

plt.show()
plt.clf()

