import logging

import bilby
import numpy as np

import memestr
from memestr.core.parameters import AllSettings


def _get_matched_filter_snrs(distances):
    matched_filter_snrs = []
    for distance in distances:
        logger = logging.getLogger('bilby')
        logger.disabled = True
        settings = AllSettings.from_defaults_with_some_specified_kwargs(luminosity_distance=distance)
        outdir = 'outdir'
        settings.waveform_data.start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration
        np.random.seed(settings.other_settings.random_seed)
        logger.info("Random seed: " + str(settings.other_settings.random_seed))

        waveform_generator = bilby.gw.WaveformGenerator(
            time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
            parameters=settings.injection_parameters.__dict__,
            waveform_arguments=settings.waveform_arguments.__dict__,
            **settings.waveform_data.__dict__)
        hf_signal = waveform_generator.frequency_domain_strain()
        ifos = [memestr.wrappers.injection_recovery.get_ifo(hf_signal, name, outdir, settings, waveform_generator,
                                                            plot=False)
                for name in settings.detector_settings.detectors]
        ifos = bilby.gw.detector.InterferometerList(ifos)
        matched_filter_snr = 0
        for ifo in ifos:
            matched_filter_snr += np.abs(ifo.meta_data['matched_filter_SNR'] ** 2)
        matched_filter_snrs.append(np.sqrt(matched_filter_snr))
    return matched_filter_snrs