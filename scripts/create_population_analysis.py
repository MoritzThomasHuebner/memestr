import numpy as np

# test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(test)
snrs = np.array([])
memory_log_bfs = np.array([])
for i in range(1):
    data = np.loadtxt('Injection_log_bfs_snr_60_{}'.format(i))
    # data = test
    print(data[0])
    snrs = np.append(snrs, data[0])
    memory_log_bfs = np.append(memory_log_bfs, data[2])
print(snrs)
print(memory_log_bfs)