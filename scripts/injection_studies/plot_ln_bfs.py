import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
from scipy.stats import kurtosis
import scipy

# matplotlib.use('Qt5Agg')
plt.style.use('paper.mplstyle')
ln_bfs = np.loadtxt('summary/memory_log_bfs_recovered.txt')
print(min(ln_bfs))
print(ln_bfs[np.where(np.abs(ln_bfs) > 0.3)])
ln_bfs = ln_bfs[np.where(np.abs(ln_bfs) < 0.8)]
# plt.figure(figsize=(8, 5))
plt.hist(ln_bfs, bins=50, density=True, label="$\ln \, \mathrm{BF}_{\mathrm{mem}}$ from Ref.~[13]")
# xs = np.linspace(-0.3, 0.3, 1000)
# ys = scipy.stats.norm.pdf(xs, loc=np.mean(ln_bfs), scale=np.std(ln_bfs))
# plt.plot(xs, ys, linewidth=3, label="Normal distribution with same variance as histogram")
# plt.hist(np.random.normal(size=200000)*np.std(ln_bfs) + np.mean(ln_bfs), bins=50, density=True, alpha=0.4, label="Normal distribution")
plt.semilogy()
plt.xlabel("$\ln \, \mathrm{BF}_{\mathrm{mem}}$")
plt.ylabel("$p(\ln \, \mathrm{BF}_{\mathrm{mem}}$)")
plt.xlim(-0.8, 0.8)
plt.ylim(0.008, 5e2)
# plt.legend()
plt.tight_layout()
plt.savefig("ln_bf_distribution.pdf")
plt.show()
plt.clf()

print(kurtosis(ln_bfs[np.where(np.abs(ln_bfs) < 0.4)]))
print(kurtosis(ln_bfs))
print(np.std(ln_bfs[np.where(np.abs(ln_bfs) < 0.4)]))
print(np.std(ln_bfs))

print()
print(len(np.where(ln_bfs < -0.1)[0]))
print(len(np.where(ln_bfs < -0.12)[0]))
print(len(np.where(np.abs(ln_bfs) > 0.12)[0]))
print(len(np.where(ln_bfs > 0.05)[0]))

print(max(ln_bfs))
print(np.argmax(ln_bfs))
print(ln_bfs)


import memestr
from scipy.special import logsumexp

log_bfs_aligned = []
log_bfs_prec = []

for event in memestr.events.events:
    log_weights = np.loadtxt(f"../O3a/{event.name}_memory_log_weights")
    log_bfs_aligned.append(logsumexp(log_weights) - np.log(len(log_weights)))
for event in memestr.events.precessing_events:
    try:
        log_weights = np.loadtxt(f"../O3a/{event.name}_2000_memory_log_weights")
        log_bfs_prec.append(logsumexp(log_weights) - np.log(len(log_weights)))
    except OSError as e:
        print(e)

plt.hist(ln_bfs, histtype='step', bins=50, density=True, label="$\ln \, \mathrm{BF}_{\mathrm{mem}}$ from Ref.~[13]")
plt.hist(log_bfs_aligned, histtype='step', bins=8, density=True, label='Aligned $\ln \, \mathrm{BF}_{\mathrm{mem}}$')
plt.hist(log_bfs_prec, histtype='step', bins=8, density=True, label='Precessing $\ln \, \mathrm{BF}_{\mathrm{mem}}$')
plt.semilogy()
plt.xlabel("$\ln \, \mathrm{BF}_{\mathrm{mem}}$")
plt.ylabel("$p(\ln \, \mathrm{BF}_{\mathrm{mem}}$)")
plt.xlim(-0.8, 0.8)
plt.ylim(0.008, 5e3)
plt.legend()
plt.tight_layout()
plt.savefig("ln_bf_distribution_test.pdf")
# plt.show()
plt.clf()