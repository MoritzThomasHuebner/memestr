import numpy as np
import json
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

snrs = np.loadtxt('snr_v_log_bf_data/snrs.txt')
mem_log_bfs_reweight = []
mem_log_bfs_reweight_err = []
mem_log_bfs_injected = []
mem_log_bfs_sampled = []
mem_log_bfs_sampled_err = []

for run_id in range(16):
    run_id = str(run_id).zfill(2)

    with open(f'snr_v_log_bf_data/results/{run_id}.json') as f:
        data = json.load(f)

    injection_bf = data['injection_bfs']['0']
    sampling_bfs = [data['sampling_bfs'][str(i)] for i in range(8)]
    reweighing_to_memory_bfs = [data['reweighing_to_memory_bfs_mem_inj'][str(i)] for i in range(8)]

    mem_log_bfs_reweight.append(np.mean(reweighing_to_memory_bfs))
    mem_log_bfs_reweight_err.append(np.std(reweighing_to_memory_bfs) / np.sqrt(len(reweighing_to_memory_bfs)))
    mem_log_bfs_sampled.append(np.mean(sampling_bfs))
    mem_log_bfs_sampled_err.append(np.std(sampling_bfs) / np.sqrt(len(sampling_bfs)))
    mem_log_bfs_injected.append(injection_bf)


matplotlib.rcParams.update({'font.size': 15})

fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.plot(snrs, mem_log_bfs_sampled, label='Nested Sampling', linestyle='None', marker="v")
ax0.plot(snrs, mem_log_bfs_reweight, label='Reweighting', linestyle='None', marker="o")
ax0.plot(snrs, mem_log_bfs_injected, label='$\ln\mathcal{L}$ at injected value')
# ax0.set_ylim(-2, 2)
ax0.set_ylabel('$\ln \mathcal{BF}_{\mathrm{mem}}$')
ax0.legend()
ax0.set_xticks([])


ax1.plot(snrs, mem_log_bfs_sampled_err, label='Nested Sampling', linestyle='None', marker="v")
ax1.plot(snrs, mem_log_bfs_reweight_err, label='Reweighting', linestyle='None', marker="o")
ax1.set_yscale('log')
ax1.set_xlabel('$\\rho_{\mathrm{mf}}$')
# ax1.set_ylim(1e-4, 1)
ax1.set_yticks([1e-4, 1e-2, 1])
ax1.set_xticks([10, 20, 30, 40, 50, 60])
ax1.set_ylabel('$\Delta \ln \mathcal{BF}_{\mathrm{mem}}$')
plt.tight_layout()
plt.savefig('snr_vs_evidence_new_data.png')
plt.show()
plt.clf()
