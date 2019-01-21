import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d


from memestr.core.population import mass_distribution_no_vt


def debug_plots():
    m_1 = np.linspace(1, 50, 100)
    qs = np.linspace(0.01, 1, 10)
    for q in qs:
        dataset = dict(m1_source=m_1, q=q)

        prob = mass_distribution_no_vt(dataset=dataset, alpha=9,
                                       mmin=6.5, mmax=50, lam=0.4,
                                       mpp=30, sigpp=5, beta=5.8, delta_m=4)
        plt.plot(m_1, prob)
        plt.semilogy()
        plt.ylim(1e-6, 10)
    plt.show()
    plt.clf()


m_1 = np.linspace(1, 50, 1000)
qs = np.linspace(0.01, 1, 1000)
probs = []
for q in qs:
    dataset = dict(m1_source=m_1, q=q)
    prob = mass_distribution_no_vt(dataset=dataset, alpha=9,
                                   mmin=6.5, mmax=50, lam=0.4,
                                   mpp=30, sigpp=5, beta=5.8, delta_m=4)
    probs.append(prob)


prob_density = interp2d(x=m_1, y=qs, z=np.array(probs))

print(prob_density)


# def sample_masses_model_a():
#     return sample_masses(alpha=0.4, m_max=41.6, m_min=5)
#
#
# def sample_masses_model_b():
#     return sample_masses(alpha=1.6, m_max=42.0, m_min=7.9)
#
#
# def sample_masses(alpha, m_max, m_min):
#     m_1 = np.linspace(1, 50, 1000)
#     m_2 = np.linspace(1, 50, 1000)
#     p_m_1 = models.prob.marginal_pdf(m1=m_1, alpha=alpha, m_min=m_min, m_max=m_max, M_max=100)
#     prior_m_1 = bilby.core.prior.Interped(xx=m_1, yy=p_m_1[0], minimum=1, maximum=50)
#     m_1_sample = prior_m_1.sample()
#     p_m_2_given_m_1 = prob.joint_pdf(m_1=m_1_sample, m_2=m_2, alpha=alpha, m_min=m_min, m_max=m_max, M_max=100)
#     prior_m_2 = bilby.core.prior.Interped(xx=m_1, yy=p_m_2_given_m_1, minimum=1, maximum=50)
#     m_2_sample = prior_m_2.sample()
#     return m_1_sample, m_2_sample

debug_plots()
# m_1, m_2 = sample_masses_model_a()
# print(m_1)
# print(m_2)
