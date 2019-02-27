import matplotlib.pyplot as plt
import logging
import numpy as np

from memestr.core.submit import get_injection_parameter_set
from memestr.core.expected_bayes_factor import calculate_expected_log_bf
from memestr.core.parameters import AllSettings

logger = logging.getLogger('bilby')
logger.disabled = True

log_bfs = np.array([])
cumulative_log_bfs = np.array([])

# log_bfs = np.loadtxt('plots/log_bfs_noise.txt')
# cumulative_log_bfs = np.cumsum(log_bfs)
# print(np.mean(log_bfs))
for id in range(10000, 10650):
    settings = AllSettings()
    params = settings.injection_parameters.__dict__
    params.update(get_injection_parameter_set(id=id))
    try:
        expected_log_bf = calculate_expected_log_bf(params, duration=16)
    except ValueError:
        continue
    print(expected_log_bf)
    print(id)
    log_bfs = np.append(log_bfs, expected_log_bf)
    cumulative_log_bfs = np.append(cumulative_log_bfs, np.sum(log_bfs))


plt.xlabel('Number of events')
plt.ylabel('log BF')
plt.plot(cumulative_log_bfs, label='cumulative log BF')
plt.axhline(y=8, xmin=0, xmax=1024, color='r', linestyle=':', label='detection threshold')
# plt.xlim(0, 500)
# plt.ylim(0, 12)
plt.legend()
plt.tight_layout()
plt.savefig('plots/expected_cumulative_log_bfs_model_c_with_noise.png')
plt.clf()

# plt.hist(log_bfs, bins=100)
# plt.hist(log_bfs, bins=np.logspace(np.log10(0.00000000001), np.log10(100), 100))
# plt.xlabel('log BF per event')
# plt.ylabel('Number of events')
# plt.loglog()
# plt.semilogy()
# plt.tight_layout()
# plt.savefig('plots/model_c_histogram_noise.png')
# plt.clf()

# np.savetxt('plots/log_bfs_noise.txt', log_bfs)
