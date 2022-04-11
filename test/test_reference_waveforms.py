import unittest
import bilby
import memestr
import numpy as np


class TestWaveforms(unittest.TestCase):

    def setUp(self):
        sampling_frequency = 2048
        duration = 4
        series = bilby.core.series.CoupledTimeAndFrequencySeries(sampling_frequency=sampling_frequency,
                                                                 duration=duration)

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

        self.xhm = memestr.waveforms.phenom.xhm.td_imrx_with_memory(
            times, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)
        self.xhm_memory = memestr.waveforms.phenom.xhm.td_imrx_memory_only(
            times, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)

        self.xhm_fd = memestr.waveforms.phenom.xhm.fd_imrx(
            frequencies, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)
        self.xhm_fast_fd = memestr.waveforms.phenom.xhm.fd_imrx_fast(
            frequencies, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)
        self.xhm_memory_fd = memestr.waveforms.phenom.xhm.fd_imrx_memory_only(
            frequencies, mass_ratio, total_mass, luminosity_distance, s13, s23, inc, phase)

        self.sur7dq4 = memestr.waveforms.nrsur7dq4.td_nr_sur_7dq4_with_memory(
            times, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)
        self.sur7dq4_memory = memestr.waveforms.nrsur7dq4.td_nr_sur_7dq4_memory_only(
            times, mass_ratio, total_mass, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)

        self.sur7dq4_fd = memestr.waveforms.nrsur7dq4.fd_nr_sur_7dq4(
            frequencies, mass_ratio, total_mass, a_1, a_2, tilt_1,
            tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)
        self.sur7dq4_memory_fd = memestr.waveforms.nrsur7dq4.fd_nr_sur_7dq4_memory_only(
            frequencies, mass_ratio, total_mass, a_1, a_2, tilt_1,
            tilt_2, phi_12, phi_jl, luminosity_distance, inc, phase)
        self.outdir = 'reference_waveforms/'
        self.modes = ['plus', 'cross']

    def test_xhm_fd_consistency(self):
        for mode in self.modes:
            self.assertTrue(np.allclose(self.xhm_fd[mode].real, self.xhm_fast_fd[mode].real))
            self.assertTrue(np.allclose(self.xhm_fd[mode].imag, self.xhm_fast_fd[mode].imag))

    def test_xhm_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}xhm_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, self.xhm[mode]))

    def test_xhm_memory_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}xhm_memory_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, self.xhm_memory[mode]))

    def test_xhm_fd_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}xhm_fd_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, np.abs(self.xhm_fd[mode])))

    def test_xhm_fast_fd_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}xhm_fd_fast_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, np.abs(self.xhm_fast_fd[mode])))

    def test_xhm_memory_fd_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}xhm_memory_fd_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, np.abs(self.xhm_memory_fd[mode])))

    def test_sur_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}sur7dq4_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, self.sur7dq4[mode]))

    def test_sur_memory_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}sur7dq4_memory_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, self.sur7dq4_memory[mode]))

    def test_sur_fd_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}sur7dq4_fd_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, np.abs(self.sur7dq4_fd[mode])))

    def test_sur_memory_fd_vs_reference(self):
        for mode in self.modes:
            ref_waveform = np.loadtxt(f'{self.outdir}sur7dq4_memory_fd_{mode}.txt')
            self.assertTrue(np.allclose(ref_waveform, np.abs(self.sur7dq4_memory_fd[mode])))
