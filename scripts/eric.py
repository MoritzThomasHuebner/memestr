import bilby
import numpy as np
import matplotlib.pyplot as plt

def const_func(x):
    return mu

def zero_func(x):
    return 0


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

length = 1
mu = 0.1
sigma = 1


# prior = bilby.gw.prior.UniformComovingVolume(minimum=0, maximum=2, name='redshift')
# plt.plot(np.linspace(0, 2, 1000), prior.prob(np.linspace(0, 2, 1000)))
# plt.show()
# integral = np.trapz(prior.prob(np.linspace(0, 2, 1000)))
# print(integral)

all_log_bfs = []
log_bfs_list_list = []
for j in range(100000):
    dist = bilby.core.prior.Gaussian(mu=mu, sigma=sigma)
    data = dist.sample(length)
    cum_log_bfs = []
    likelihood_1 = bilby.core.likelihood.GaussianLikelihood(x=np.zeros(1), y=data, func=const_func, sigma=sigma)
    likelihood_2 = bilby.core.likelihood.GaussianLikelihood(x=np.zeros(1), y=data, func=zero_func, sigma=sigma)
    log_bf = likelihood_1.log_likelihood() - likelihood_2.log_likelihood()
    all_log_bfs.append(log_bf)
    log_bfs_list_list.append(cum_log_bfs)
    # print(j)

# for cum_log_bfs in log_bfs_list_list:
#     plt.plot(cum_log_bfs, alpha=0.3, color='grey')
# plt.show()
# plt.clf()


plt.hist(np.array(all_log_bfs), bins='fd', normed=True)
plt.plot(np.linspace(-1, 1, 100), gaussian(np.linspace(-1, 1, 100), np.mean(all_log_bfs), np.std(all_log_bfs)), color='red')
# plt.semilogy()
plt.show()
plt.clf()
# snr = 1/length * np.sum(data)
# print(snr)