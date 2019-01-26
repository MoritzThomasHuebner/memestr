import bilby as bb
import memestr as me
import logging


def expected_bayes_factor(luminosity_distances=None, distance_marginalization=False, **params):
    settings = me.core.parameters.AllSettings()
    parameters = settings.injection_parameters.__dict__
    parameters.update(params)
    priors = settings.recovery_priors.proper_dict()

    sampling_frequency = settings.waveform_data.sampling_frequency
    duration = settings.waveform_data.duration
    start_time = settings.waveform_data.start_time

    logger = logging.getLogger('bilby')
    logger.disabled = True

    if not luminosity_distances:
        luminosity_distances = [settings.injection_parameters.luminosity_distance]

    log_bfs = []
    for luminosity_distance in luminosity_distances:
        parameters['luminosity_distance'] = luminosity_distance
        waveform_generator_with_mem = bb.gw.WaveformGenerator(
            time_domain_source_model=me.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time)
        waveform_generator_no_mem = bb.gw.WaveformGenerator(
            time_domain_source_model=me.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time)

        ifos = [bb.gw.detector.get_interferometer_with_fake_noise_and_injection(
            name=name,
            injection_parameters=parameters,
            waveform_generator=waveform_generator_with_mem,
            sampling_frequency=sampling_frequency,
            duration=duration, start_time=start_time,
            zero_noise=True,
            plot=False) for name in ['H1', 'L1', 'V1']]

        likelihood_with_mem = \
            bb.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                        waveform_generator=waveform_generator_with_mem,
                                                        priors=priors,
                                                        distance_marginalization=distance_marginalization)
        likelihood_no_mem = \
            bb.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                        waveform_generator=waveform_generator_no_mem,
                                                        priors=priors,
                                                        distance_marginalization=distance_marginalization)
        likelihood_with_mem.parameters = parameters
        likelihood_no_mem.parameters = parameters

        log_bf = likelihood_with_mem.log_likelihood() - likelihood_no_mem.log_likelihood()
        print(str(luminosity_distance) + '\t:' + str(log_bf))
        log_bfs.append(log_bf)

    return luminosity_distances, log_bfs
