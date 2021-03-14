from pathlib import Path
import shutil

import memestr
import bilby
import numpy as np
import matplotlib.pyplot as plt

sampling_frequency = 2048
duration = 4
series = bilby.core.series.CoupledTimeAndFrequencySeries(sampling_frequency=sampling_frequency, duration=duration)

times = series.time_array
frequencies = series.frequency_array

mass_ratio = 0.8
total_mass = 60.

luminosity_distance = 500
inc = 1.3
phase = 1.5

a_1 = 0.1
a_2 = 0.3
tilt_1 = 0.6
tilt_2 = 0.2
phi_12 = 0.4
phi_jl = 0.6

s13 = 0.1
s23 = 0.4


xhm = memestr.waveforms.phenom.xhm.td_imrx_with_memory(times, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)
xhm_memory = memestr.waveforms.phenom.xhm.td_imrx_memory_only(times, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)

xhm_fd = memestr.waveforms.phenom.xhm.fd_imrx(frequencies, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)
xhm_fd_fast = memestr.waveforms.phenom.xhm.fd_imrx_fast(frequencies, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)
xhm_memory_fd = memestr.waveforms.phenom.xhm.fd_imrx_memory_only(frequencies, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)

sur7dq4 = memestr.waveforms.nrsur7dq4.td_nr_sur_7dq4_with_memory(times, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)
sur7dq4_memory = memestr.waveforms.nrsur7dq4.td_nr_sur_7dq4_memory_only(times, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)

sur7dq4_fd = memestr.waveforms.nrsur7dq4.fd_nr_sur_7dq4(frequencies, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)
sur7dq4_memory_fd = memestr.waveforms.nrsur7dq4.fd_nr_sur_7dq4_memory_only(frequencies, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)

outdir = 'reference_waveforms/'
shutil.rmtree(outdir, ignore_errors=True)

Path(outdir).mkdir(parents=True, exist_ok=True)

for mode in ['plus', 'cross']:
    plt.figure(dpi=150)
    plt.plot(times, xhm[mode], label='XHM')
    plt.plot(times, xhm_memory[mode], label='XHM memory')
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('h')
    plt.tight_layout()
    plt.savefig(f'{outdir}XHM_{mode}.pdf')
    plt.clf()

    plt.figure(dpi=150)
    plt.plot(times, sur7dq4[mode], label='Sur')
    plt.plot(times, sur7dq4_memory[mode], label='Sur memory')
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('h')
    plt.tight_layout()
    plt.savefig(f'{outdir}Sur_{mode}.pdf')
    plt.clf()

    plt.figure(dpi=150)
    plt.loglog(frequencies, np.abs(xhm_fd[mode]), label='XHM')
    plt.loglog(frequencies, np.abs(xhm_memory_fd[mode]), label='XHM memory')
    plt.loglog(frequencies, np.abs(xhm_fd_fast[mode]), label='XHM fast')
    plt.legend()
    plt.xlabel('frequencies [Hz]')
    plt.ylabel('h')
    plt.xlim(20, 1024)
    plt.tight_layout()
    plt.savefig(f'{outdir}XHM_fd_{mode}.pdf')
    plt.clf()

    plt.figure(dpi=150)
    plt.loglog(frequencies, np.abs(sur7dq4_fd[mode]), label='Sur')
    plt.loglog(frequencies, np.abs(sur7dq4_memory_fd[mode]), label='Sur memory')
    plt.legend()
    plt.xlabel('frequencies [Hz]')
    plt.ylabel('h')
    plt.xlim(20, 1024)
    plt.tight_layout()
    plt.savefig(f'{outdir}Sur_fd_{mode}.pdf')
    plt.clf()

    np.savetxt(f'{outdir}xhm_{mode}.txt', xhm[mode])
    np.savetxt(f'{outdir}xhm_fd_{mode}.txt', np.abs(xhm_fd[mode]))
    np.savetxt(f'{outdir}xhm_fd_fast_{mode}.txt', np.abs(xhm_fd_fast[mode]))
    np.savetxt(f'{outdir}xhm_memory_{mode}.txt', xhm_memory[mode])
    np.savetxt(f'{outdir}xhm_memory_fd_{mode}.txt', np.abs(xhm_memory_fd[mode]))
    np.savetxt(f'{outdir}sur7dq4_{mode}.txt', sur7dq4[mode])
    np.savetxt(f'{outdir}sur7dq4_fd_{mode}.txt', np.abs(sur7dq4_fd[mode]))
    np.savetxt(f'{outdir}sur7dq4_memory_{mode}.txt', sur7dq4_memory[mode])
    np.savetxt(f'{outdir}sur7dq4_memory_fd_{mode}.txt', np.abs(sur7dq4_memory_fd[mode]))
