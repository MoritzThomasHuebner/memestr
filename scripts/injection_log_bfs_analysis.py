import numpy as np
import matplotlib.pyplot as plt

log_bfs = np.array([])
trials = np.array([])

for i in range(32):
    data = np.loadtxt('Injection_log_bfs/Injection_log_bfs_{}.txt'.format(i))
    log_bfs = np.append(log_bfs, data[:, 0])
    trials = np.append(trials, data[:, 1])

print(len(log_bfs))
print(len(trials))

print(np.mean(log_bfs))
print(np.mean(trials))

print(np.max(log_bfs))

plt.plot(np.cumsum(log_bfs))
plt.xlabel('Number of events')
plt.ylabel('')
plt.show()
plt.clf()

plt.hist(log_bfs, bins=int(np.sqrt(len(log_bfs))))
plt.show()
plt.clf()

plt.hist(trials, bins=int(np.sqrt(len(log_bfs))))
plt.show()
plt.clf()
#
# minimums = []
#
# for i in range(1000000):
#     print(i)
#     np.random.shuffle(log_bfs)
#     minimums.append(np.min(np.cumsum(log_bfs)))
minimums = np.loadtxt('minimums.txt')
percentiles = np.percentile(minimums, [100*(1-0.9999994), 100*(1-0.99994), 0.3, 5, 32])
plt.hist(minimums, alpha=0.5, bins=1000)
plt.xlim(5, -12)
colors = ['red', 'orange', 'cyan', 'black', 'green']
for i in range(len(percentiles)):
    plt.axvline(percentiles[i], label='${}\sigma$'.format(5 - i), color=colors[i])
plt.legend()
plt.xlabel('Minimum log BF')
plt.ylabel('Count')
plt.semilogy()
plt.show()
plt.clf()

# np.savetxt('minimums.txt', np.array(minimums))
