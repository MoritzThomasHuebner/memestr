import pickle
import sys
import numpy as np
from scipy.special import logsumexp

import bilby
from copy import deepcopy
import memestr
from scipy.optimize import minimize

time_tags = ["1126259462-391", "1128678900-4", "1135136350-6", "1167559936-6", "1180922494-5", "1185389807-3",
             "1186302519-7", "1186741861-5", "1187058327-1", "1187529256-5", "1239082262-222168"]
events = ["GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
          "GW170809", "GW170814", "GW170818", "GW170823", "GW190412"]



log_bfs = []
for event in events:
    log_weights = np.loadtxt("{}_log_weights".format(event))
    reweighted_log_bf = logsumexp(log_weights) - np.log(len(log_weights))
    print(reweighted_log_bf)
    log_bfs.append(reweighted_log_bf)
#
gwtm_1_original = np.loadtxt("GWTM-1.txt")
#
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
import matplotlib.pyplot as plt
#
print(np.sum(log_bfs))
plt.scatter(np.arange(0, 11), log_bfs)
#plt.scatter(np.arange(0, 10), gwtm_1_original, label="Huebner et al. (NRHybSur)")
plt.xticks(ticks=np.arange(0, 11), labels=events, rotation=75)
plt.ylabel("ln BF")
#plt.legend()
plt.tight_layout()
plt.savefig("gwtm-1_new.png")
plt.clf()
