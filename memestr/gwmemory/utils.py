from __future__ import division, absolute_import, print_function

import numpy as np

from memestr.gwmemory import harmonics


def zero_pad_time_series(times, mode):
    required_zeros = len(times) - len(mode)
    result = np.zeros(times.shape, dtype=np.complex128)
    if required_zeros > 0:
        result[:mode.shape[0]] = mode
        return result
    elif required_zeros < 0:
        return mode[-times.shape[0]:]
    else:
        return mode


def combine_modes(h_lm, inc, phase):
    """Calculate the plus and cross polarisations of the waveform from the spherical harmonic decomposition."""
    total = sum([h_lm[(l, m)] * harmonics.sYlm(-2, l, m, inc, phase) for l, m in h_lm])
    return dict(plus=total.real, cross=-total.imag)