import numpy as np
import bilby as bb
from memestr.core.parameters import AllSettings
from memestr.core.waveforms import time_domain_nr_sur_waveform_with_memory, time_domain_IMRPhenomD_waveform_with_memory
import copy

outdir = 'posterior_reweighting'

result = bb.result.read_in_result(outdir=outdir, label='IMR_mem_inj_mem_rec')
reweighted_result = copy.deepcopy(result)
print(result)

start_time = -0.5
duration = 0.5

settings = AllSettings.from_defaults_with_some_specified_kwargs(duration=duration,
                                                                start_time=start_time)

waveform_generator = bb.gw.WaveformGenerator(time_domain_source_model=time_domain_nr_sur_waveform_with_memory,
                                             parameters=settings.injection_parameters.__dict__,
                                             waveform_arguments=settings.waveform_arguments.__dict__,
                                             **settings.waveform_data.__dict__)

waveform_generator_injection = bb.gw.WaveformGenerator(time_domain_source_model=time_domain_IMRPhenomD_waveform_with_memory,
                                                       parameters=settings.injection_parameters.__dict__,
                                                       waveform_arguments=settings.waveform_arguments.__dict__,
                                                       **settings.waveform_data.__dict__)

hf_signal = waveform_generator_injection.frequency_domain_strain()

ifos = [bb.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name,
    injection_polarizations=hf_signal,
    injection_parameters=settings.injection_parameters.__dict__,
    outdir=outdir,
    zero_noise=settings.detector_settings.zero_noise,
    **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors]

priors = settings.recovery_priors.proper_dict()
likelihood = bb.gw.likelihood \
    .GravitationalWaveTransient(interferometers=ifos,
                                waveform_generator=waveform_generator,
                                priors=priors,
                                time_marginalization=settings.other_settings.time_marginalization,
                                distance_marginalization=settings.other_settings.distance_marginalization,
                                phase_marginalization=settings.other_settings.phase_marginalization)



result.plot_corner()
