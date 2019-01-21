from __future__ import division, print_function

import numpy as np
import bilby
import memestr
import logging
import copy

settings = memestr.core.parameters.AllSettings()

outdir = 'optimal_event'
label = 'optimal_event_snr'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(88170235)


class OptimalSNRLikelihood(bilby.core.likelihood.Likelihood):

    def __init__(self, interferometers, waveform_generator=None):
        super().__init__(dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.waveform_generator.sampling_frequency,
            duration=self.waveform_generator.duration,
            start_time=self.waveform_generator.start_time)

    def log_likelihood(self):
        ifos = copy.deepcopy(self.interferometers)
        logger = logging.getLogger('bilby')
        logger.disabled = True
        ifos.inject_signal(waveform_generator=self.waveform_generator,
                           parameters=self.parameters)
        logger.disabled = False
        return 100*np.sqrt(np.sum([ifo.meta_data['optimal_SNR'] ** 2 for ifo in ifos]))


priors = settings.recovery_priors.proper_dict()
for key in ['phase', 'luminosity_distance', 'geocent_time', 'total_mass', 'mass_ratio',
            's11', 's12', 's13', 's21', 's22', 's23']:
    priors[key] = settings.injection_parameters.__dict__[key]

wg = bilby.gw.WaveformGenerator(
    duration=settings.waveform_data.duration, sampling_frequency=settings.waveform_data.sampling_frequency,
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
    waveform_arguments=settings.waveform_arguments.__dict__, parameters=settings.injection_parameters.__dict__)


likelihood = OptimalSNRLikelihood(
    interferometers=bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']),
    waveform_generator=wg)

result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500, outdir=outdir,
                           label=label, sample='unif')

result.plot_corner()
