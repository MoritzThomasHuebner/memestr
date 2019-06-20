import numpy as np
from scipy.interpolate import interp1d
from scipy.special import i0e, logsumexp
from gwmemory.waveforms import combine_modes

from bilby.gw.likelihood import GravitationalWaveTransient


class HOMTimePhaseMarginalizedGWT(GravitationalWaveTransient):

    def __init__(self, interferometers, waveform_generator, priors):
        super(HOMTimePhaseMarginalizedGWT, self).__init__(interferometers, waveform_generator,
                                                          time_marginalization=True,
                                                          distance_marginalization=False,
                                                          phase_marginalization=True, priors=priors,
                                                          distance_marginalization_lookup_table=None)

    def log_likelihood_ratio(self):
        waveform_polarizations = self.waveform_generator.frequency_domain_strain(self.parameters)

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        d_inner_h = 0.
        optimal_snr_squared = 0.
        complex_matched_filter_snr = 0.
        d_inner_h_tc_array = np.zeros(
            self.interferometers.frequency_array[0:-1].shape,
            dtype=np.complex128)
        for mode in waveform_polarizations:
            ll, mm = mode[0], mode[1]
            for interferometer in self.interferometers:
                waveform_polarizations_plus_cross = combine_modes(h_lm={(ll, mm): waveform_polarizations[mode]},
                                                                  inc=self.parameters['inc'],
                                                                  phase=self.parameters['phase'])
                per_detector_snr = self.calculate_snrs(
                    waveform_polarizations=waveform_polarizations_plus_cross,
                    interferometer=interferometer)

                d_inner_h += per_detector_snr.d_inner_h
                optimal_snr_squared += np.real(per_detector_snr.optimal_snr_squared)
                complex_matched_filter_snr += per_detector_snr.complex_matched_filter_snr
                d_inner_h_tc_array += per_detector_snr.d_inner_h_squared_tc_array * np.exp(
                    1j * mm * self.parameters['phase'])

        log_l_tc_array = np.sum(d_inner_h_tc_array.real) * self.delta_tc - optimal_snr_squared / 2
        log_l = logsumexp(log_l_tc_array, b=self.time_prior_array)
        return float(log_l.real)

    def _setup_time_marginalization(self):
        self.delta_tc = 2 / self.waveform_generator.sampling_frequency
        super(HOMTimePhaseMarginalizedGWT, self)._setup_time_marginalization()

    def _setup_phase_marginalization(self):
        pass

    def _setup_distance_marginalization(self, lookup_table=None):
        pass

    def calculate_snrs(self, waveform_polarizations, interferometer):
        """
        Compute the snrs

        Parameters
        ----------
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object

        """
        signal = interferometer.get_detector_response(
            waveform_polarizations, self.parameters)
        d_inner_h = interferometer.inner_product(signal=signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared ** 0.5)

        d_inner_h_squared_tc_array = \
            4 / self.waveform_generator.duration * np.fft.fft(
                signal[0:-1] *
                interferometer.frequency_domain_strain.conjugate()[0:-1] /
                interferometer.power_spectral_density_array[0:-1])

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=d_inner_h_squared_tc_array)
