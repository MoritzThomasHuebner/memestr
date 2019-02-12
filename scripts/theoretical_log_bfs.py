import matplotlib.pyplot as plt
import logging
import numpy as np

from memestr.core.submit import get_injection_parameter_set
from memestr.core.expected_bayes_factor import calculate_expected_log_bf
from memestr.core.parameters import AllSettings

logger = logging.getLogger('bilby')
logger.disabled = True

log_bfs = []
cummulative_log_bfs = []

for id in range(256):
    settings = AllSettings()
    params = settings.injection_parameters.__dict__
    params.update(get_injection_parameter_set(id=id))
    try:
        expected_log_bf = calculate_expected_log_bf(params, duration=16)
    except ValueError:
        continue
    print(expected_log_bf)
    print(id)
    log_bfs.append(expected_log_bf)
    cummulative_log_bfs.append(np.sum(log_bfs))

plt.plot(cummulative_log_bfs)
plt.xlabel('Number of events')
plt.ylabel('Cumulative log BF')
plt.savefig('plots/expected_cumulative_log_bfs_model_b.png')
plt.clf()
