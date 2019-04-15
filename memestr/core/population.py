from __future__ import division

import numpy as np
from collections import namedtuple
import random

import bilby
from bilby.gw import conversion
import os
import gwpopulation
import matplotlib.pyplot as plt

MassContainer = namedtuple('MassContainer', ['primary_masses', 'secondary_masses',
                                             'mass_ratios', 'total_masses', 'chirp_masses'])
SpinContainer = namedtuple('SpinContainer', ['s13', 's23'])
ExtrinsicParameterContainer = namedtuple('ExtrinisicParamterContainer', ['inc', 'ra', 'dec',
                                                                         'phase', 'psi', 'geocent_time',
                                                                         'luminosity_distance'])
AllParameterContainer = namedtuple('AllParameterContainer',
                                   ['primary_masses', 'secondary_masses', 'mass_ratios', 'total_masses',
                                    'chirp_masses', 's13', 's23', 'inc', 'ra', 'dec',
                                    'phase', 'psi', 'geocent_time', 'luminosity_distance'])


def generate_mass_parameters(size=10000, clean=False, alpha=1.5, mmin=8, mmax=45, beta=3, plot=False):
    m1s = np.linspace(4, 45, size)
    qs = np.linspace(0.01, 1, size)
    q_mesh, m_mesh = np.meshgrid(qs, m1s)

    outfile = 'pop_masses_{}.txt'.format(size)

    if clean or not os.path.isfile(outfile):
        primary_masses, mass_ratios = \
            _generate_masses(m_mesh, q_mesh, size, alpha=alpha, m_min=mmin, m_max=mmax, beta=beta)
        save = np.array((primary_masses, mass_ratios))
        np.savetxt(outfile, save)
    else:
        pop_masses = np.loadtxt(outfile)
        primary_masses = pop_masses[0]
        mass_ratios = pop_masses[1]
    secondary_masses = primary_masses * mass_ratios
    total_masses = primary_masses + secondary_masses
    chirp_masses = conversion.component_masses_to_chirp_mass(primary_masses, secondary_masses)
    if plot:
        mass_debug_plots(mass_ratios, primary_masses, secondary_masses, total_masses, chirp_masses)
    return MassContainer(primary_masses=primary_masses, secondary_masses=secondary_masses,
                         mass_ratios=mass_ratios, total_masses=total_masses, chirp_masses=chirp_masses)


def _generate_masses(m_mesh, q_mesh, size, alpha, m_min, m_max, beta):
    dataset = dict(mass_1=m_mesh, mass_ratio=q_mesh)
    weights = gwpopulation.models.mass.power_law_primary_mass_ratio(dataset=dataset, alpha=alpha,
                                                                    mmin=m_min, mmax=m_max, beta=beta)
    norm_weights = weights / np.max(weights)
    random_numbers = np.random.random(size=(size, size))
    valid_samples = random_numbers < norm_weights
    primary_masses_filtered = []
    mass_ratios_filtered = []
    for i in range(len(weights[0])):
        for j in range(len(weights[:, 0])):
            if valid_samples[i][j]:
                primary_masses_filtered.append(m_mesh[i][j])
                mass_ratios_filtered.append(q_mesh[i][j])
    # if len(primary_masses_filtered) > size:
    #     primary_masses_filtered, mass_ratios_filtered = \
    #         random.sample(zip(primary_masses_filtered, mass_ratios_filtered), size)
    # elif len(primary_masses_filtered) < size:
    #     raise ValueError('Insufficient Samples.')

    primary_masses_filtered = np.array(primary_masses_filtered)
    mass_ratios_filtered = np.array(mass_ratios_filtered)
    return np.array(primary_masses_filtered), np.array(mass_ratios_filtered)


def mass_debug_plots(mass_ratios, primary_masses, secondary_masses, total_masses, chirp_masses):
    plt.scatter(primary_masses, mass_ratios)
    plt.xlabel('Primary mass')
    plt.ylabel('Mass ratio')
    plt.show()
    plt.clf()

    _debug_histogram(primary_masses, 'Primary mass')
    _debug_histogram(secondary_masses, 'Secondary mass')
    _debug_histogram(mass_ratios, 'Mass ratio')
    _debug_histogram(total_masses, 'Total Mass')
    _debug_histogram(chirp_masses, 'Chirp mass')


def generate_spins(size=10000, plot=False):
    prior = bilby.gw.prior.AlignedSpin(name='s13', a_prior=bilby.core.prior.Uniform(0.0, 0.5), latex_label='s13')
    s13 = prior.sample(size)
    s23 = prior.sample(size)
    if plot:
        spin_debug_plot(s13)
    return SpinContainer(s13=np.array(s13), s23=np.array(s23))


def spin_debug_plot(spins):
    _debug_histogram(spins, 'Spin', log=False)


def _debug_histogram(parameter, name, log=True):
    plt.hist(parameter, bins=int(np.sqrt(len(parameter))))
    if log:
        plt.semilogy()
    plt.xlabel(name)
    plt.show()
    plt.clf()


def generate_extrinsic_parameters(size=10000, plot=False):
    priors_inc = bilby.core.prior.Sine(latex_label="$\\theta_{jn}$")
    priors_ra = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi)
    priors_dec = bilby.core.prior.Cosine()
    priors_phase = bilby.core.prior.Uniform(minimum=0, maximum=np.pi)
    priors_psi = bilby.core.prior.Uniform(minimum=0, maximum=np.pi/2)
    priors_geocent_time = bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)
    priors_luminosity_distance = bilby.gw.prior.UniformComovingVolume(minimum=10, maximum=5000,
                                                                      name='luminosity_distance')
    inc = priors_inc.sample(size=size)
    ra = priors_ra.sample(size=size)
    dec = priors_dec.sample(size=size)
    phase = priors_phase.sample(size=size)
    psi = priors_psi.sample(size=size)
    geocent_time = priors_geocent_time.sample(size=size)
    luminosity_distance = priors_luminosity_distance.sample(size=size)
    if plot:
        extrinsic_parameters_debug_plots(inc=inc, ra=ra, dec=dec, phase=phase, psi=psi,
                                         geocent_time=geocent_time, luminosity_distance=luminosity_distance)
    geocent_time = priors_geocent_time.sample(size=size)

    return ExtrinsicParameterContainer(inc=inc, ra=ra, dec=dec, phase=phase, psi=psi,
                                       geocent_time=geocent_time, luminosity_distance=luminosity_distance)


def extrinsic_parameters_debug_plots(inc, ra, dec, phase, psi, geocent_time, luminosity_distance):
    _debug_histogram(inc, 'Inclination', log=False)
    _debug_histogram(ra, 'RA', log=False)
    _debug_histogram(dec, 'DEC', log=False)
    _debug_histogram(phase, '$\phi$', log=False)
    _debug_histogram(psi, '$\psi$', log=False)
    _debug_histogram(geocent_time, 'Time of Coalescence', log=False)
    _debug_histogram(luminosity_distance, 'Luminosity_distance', log=True)


def generate_all_parameters(size=10000, plot=False, **mass_kwargs):
    mps = generate_mass_parameters(size=size, plot=plot, **mass_kwargs)
    sps = generate_spins(size=size, plot=plot)
    eps = generate_extrinsic_parameters(size=size, plot=plot)
    return AllParameterContainer(primary_masses=mps.primary_masses, secondary_masses=mps.secondary_masses,
                                 total_masses=mps.total_masses, mass_ratios=mps.mass_ratios,
                                 chirp_masses=mps.chirp_masses, s13=sps.s13, s23=sps.s23,
                                 inc=eps.inc, ra=eps.ra, dec=eps.dec,
                                 psi=eps.psi, phase=eps.phase, geocent_time=eps.geocent_time,
                                 luminosity_distance=eps.luminosity_distance)
