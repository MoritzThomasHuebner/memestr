import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.stats
import matplotlib.pyplot as plt
import sys
# label = sys.argv[1]
label = 'kagra'
# label = 'snr_8_12_mem'

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

events_to_detection = 8*len(log_bfs)/np.sum(log_bfs)
events_per_day = 53.2/365 * 4/3*np.pi*5**3
print("Memory log BF per Event: " + str(np.mean(log_bfs)))
print("Standard deviation log BF: " + str(np.std(log_bfs)))
print("Events to log BF = 8: " + str(events_to_detection))
print("Total number of events considered: " + str(len(log_bfs)))
print("Total number of events drawn: " + str(np.sum(trials)))
print("Fraction of detected events: " + str(1/np.mean(trials)))
print("Total number of BBH events in prior volume to log BF = 8: " + str(events_to_detection*np.mean(trials)))
print("Number of events per day in prior volume " + str(events_per_day))
print("Number of duty cycle days to detection: " + str(events_to_detection*np.mean(trials)/events_per_day))

# low_snr_log_bf = np.sum(log_bfs[np.where(snrs < 16)])
# high_snr_log_bf = np.sum(log_bfs[np.where(snrs > 16)])
# print(low_snr_log_bf)
# print(high_snr_log_bf)

# sys.exit(0)
# plt.hist(log_bfs, bins='fd')
# plt.semilogy()
# plt.show()
# plt.clf()
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

required_events = []
for i in range(20000):
    print(i)
    tot = 0
    j = 0
    while tot < 8:
        tot += np.sum(np.random.choice(log_bfs, 10))
        j += 10
    required_events.append(j)
print(np.mean(required_events))
print(np.median(required_events))
print(np.percentile(required_events, [5, 95]))
plt.hist(required_events, bins='fd')
plt.show()
plt.clf()
sys.exit(0)
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
# sys.exit(0)

maximums = []
for i in range(1000000):
    realization = -np.random.choice(log_bfs, 10)
    realization_cumsum = np.cumsum(realization)
    maximums.append(np.max(realization_cumsum))
# np.savetxt('Injection_log_bfs/{}_maximums.txt'.format(label), maximums)
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

percentiles = np.percentile(maximums, np.linspace(0, 100, 100000))
false_alarm_prob = 1 - np.linspace(0, 1, 100000)
from scipy.interpolate import interp1d
# threshold_false_alarm = interp1d(percentiles, false_alarm_prob)(8)
# current_false_alarm = interp1d(percentiles, false_alarm_prob)(0.003)
plt.plot(percentiles, false_alarm_prob)
# plt.axvline(8, color='red', linestyle='--')
# plt.axhline(threshold_false_alarm, color='red', linestyle='--', label='Threshold False Alarm Rate')
# plt.ylim(0, 100)
plt.xlim(0, 13)
plt.semilogy()
plt.xlabel('log BF')
plt.ylabel('log False Alarm Probability')
plt.show()

# print('Threshold False Alarm Rate: ' + str(threshold_false_alarm))
# print('Current False Alarm Rate: ' + str(current_false_alarm))