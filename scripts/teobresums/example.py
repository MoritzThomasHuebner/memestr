import matplotlib.pyplot as plt
import numpy as np

import bilby
from memestr.waveforms.teobresums import *


mass_ratio = 0.8
total_mass = 65.
chi_1 = 0.
chi_2 = 0.
luminosity_distance = 500.
l_max = 2
phase = 0.1
inc = np.pi/2
minimum_frequency = 20.
ecc = 0.3

series = bilby.core.series.CoupledTimeAndFrequencySeries(duration=4, sampling_frequency=2048)
times = series.time_array
frequencies = series.frequency_array

waveform = fd_teob_memory_only(frequencies=frequencies, mass_ratio=mass_ratio, total_mass=total_mass, chi_1=chi_1,
                               chi_2=chi_2, luminosity_distance=luminosity_distance, inc=inc, phase=phase, ecc=ecc,
                               modes=[[2, 2]], minimum_frequency=20)

plt.loglog(frequencies, np.abs(waveform['plus']))
plt.xlim(10, 1024)
plt.savefig('example_teob.png')
plt.clf()
