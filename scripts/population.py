import numpy as np
import matplotlib.pyplot as plt

from population import models


def debug_plots():
    m_1 = np.linspace(1, 50, 1000)
    m_2 = np.linspace(1, 50, 1000)

    dataset = dict(m1_source=m_1, m2_source=m_2)

    a = models.mass_distribution_no_vt(dataset=dataset, alpha=0.4,
                                                  mmin=5, mmax=41.6, lam=0,
                                                  mpp=0, sigpp=0, beta=0, delta_m=0)
    plt.plot(m_1, a)
    plt.semilogy()
    plt.show()
    plt.clf()


def sample_masses_model_a():
    return sample_masses(alpha=0.4, m_max=41.6, m_min=5)


def sample_masses_model_b():
    return sample_masses(alpha=1.6, m_max=42.0, m_min=7.9)


# def sample_masses(alpha, m_max, m_min):
#     m_1 = np.linspace(1, 50, 1000)
#     m_2 = np.linspace(1, 50, 1000)
#     p_m_1 = prob.marginal_pdf(m1=m_1, alpha=alpha, m_min=m_min, m_max=m_max, M_max=100)
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
