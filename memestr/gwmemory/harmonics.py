import lal
import numpy as np


def sYlm(ss: int, ll: int, mm: int, theta: float, phase: float) -> complex:
    """

    Parameters
    ----------
    ss: The spin weight should be -2 for gravitational waves.
    ll: The l-mode number.
    mm: The m-mode number.
    theta: The inclination angle.
    phase: The phase.

    Returns
    -------
    The spin-weighted spherical harmonic.
    """
    return lal.SpinWeightedSphericalHarmonic(theta, np.pi - phase, ss, ll, mm)


def lmax_modes(lmax):
    """Compute all (l, m) pairs with 2<=l<=lmax"""
    return [(ll, m) for ll in range(2, lmax+1) for m in range(-ll, ll+1)]
