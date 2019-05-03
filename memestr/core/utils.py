import logging
from copy import deepcopy

import bilby
import numpy as np
from matplotlib import pyplot as plt

import memestr
from memestr.core.parameters import AllSettings


def _get_matched_filter_snrs(distances, model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory):
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
            time_domain_source_model=model,
            parameters=settings.injection_parameters.__dict__,
            waveform_arguments=settings.waveform_arguments.__dict__,
            **settings.waveform_data.__dict__)
        hf_signal = waveform_generator.frequency_domain_strain()
        ifos = [get_ifo(hf_signal, name, outdir, settings, waveform_generator,
                        plot=False)
                for name in settings.detector_settings.detectors]
        ifos = bilby.gw.detector.InterferometerList(ifos)
        matched_filter_snr = 0
        for ifo in ifos:
            matched_filter_snr += np.abs(ifo.meta_data['matched_filter_SNR'] ** 2)
        matched_filter_snrs.append(np.sqrt(matched_filter_snr))
    return matched_filter_snrs


def get_ifo(hf_signal, name, outdir, settings, waveform_generator, label='', plot=True):
    interferometer = bilby.gw.detector.get_empty_interferometer(name)
    if name in ['H1', 'L1']:
        interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()
    elif name in ['V1']:
        interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity. \
            from_power_spectral_density_file('AdV_psd.txt')
    if settings.detector_settings.zero_noise:
        interferometer.set_strain_data_from_zero_noise(**settings.waveform_data.__dict__)
    else:
        interferometer.set_strain_data_from_power_spectral_density(**settings.waveform_data.__dict__)
    injection_polarizations = interferometer.inject_signal(
        parameters=deepcopy(settings.injection_parameters.__dict__),
        injection_polarizations=hf_signal,
        waveform_generator=waveform_generator)
    signal = interferometer.get_detector_response(
        injection_polarizations, settings.injection_parameters.__dict__)
    if plot:
        plot_ifo(interferometer, signal=signal, outdir=outdir, label=label)
    interferometer.save_data(outdir)
    return interferometer


def plot_ifo(ifo, signal=None, outdir='.', label=None):
    from bilby import utils
    import bilby.gw.utils as gwutils
    if utils.command_line_args.test:
        return
    fig, ax = plt.subplots()
    ax.loglog(ifo.frequency_array,
              gwutils.asd_from_freq_series(freq_data=ifo.frequency_domain_strain,
                                           df=(ifo.frequency_array[1] - ifo.frequency_array[0])),
              color='C0', label=ifo.name)
    ax.loglog(ifo.frequency_array,
              ifo.amplitude_spectral_density_array,
              color='C1', lw=0.5, label=ifo.name + ' ASD')
    if signal is not None:
        ax.loglog(ifo.frequency_array,
                  gwutils.asd_from_freq_series(freq_data=signal,
                                               df=(ifo.frequency_array[1] - ifo.frequency_array[0])),
                  color='C2',
                  label='Signal')
    ax.grid(True)
    ax.set_ylabel(r'strain [strain/$\sqrt{\rm Hz}$]')
    ax.set_xlabel(r'frequency [Hz]')
    ax.set_xlim(20, 2000)
    ax.set_ylim(1e-33, 1e-21)
    plt.tight_layout()
    ax.legend(loc='best')
    if label is None:
        fig.savefig(
            '{}/{}_frequency_domain_data.png'.format(outdir, ifo.name))
    else:
        fig.savefig(
            '{}/{}_{}_frequency_domain_data.png'.format(
                outdir, ifo.name, label))