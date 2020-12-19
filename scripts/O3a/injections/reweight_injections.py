import sys

import bilby
import numpy as np
from scipy.special import logsumexp

import memestr

duration = 4.
sampling_frequency = 2048.

modes = ['aligned', 'precessing']
mode = modes[int(sys.argv[1])]
run_id = int(sys.argv[2])

outdir = f'outdir_{mode}_injections'
label = f'run_{run_id}'
bilby.core.utils.setup_logger(outdir=outdir, label=label)


mass_1 = 40.
mass_2 = 20
total_mass = mass_1 + mass_2
mass_ratio = mass_2 / mass_1

if mode == 'aligned':
    injection_parameters = dict(
        mass_1=mass_1,
        mass_2=mass_2,
        a_1=0.3,
        a_2=0.1,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=-0.0,
        phi_jl=0.0,
        luminosity_distance=400,
        dec=-0.5,
        ra=2.4,
        theta_jn=1.5,
        psi=0.0,
        phase=0.0,
        geocent_time=0.0
    )
else:
    injection_parameters = dict(
        mass_1=mass_1,
        mass_2=mass_2,
        a_1=0.3,
        a_2=0.1,
        tilt_1=-0.4,
        tilt_2=0.3,
        phi_12=-0.1,
        phi_jl=0.2,
        luminosity_distance=400,
        dec=-0.5,
        ra=2.4,
        theta_jn=1.5,
        psi=0.0,
        phase=0.0,
        geocent_time=0.0
    )


waveform_arguments_injection = dict(waveform_approximant='IMRPhenomPv3HM',
                                    reference_frequency=50., minimum_frequency=20.)
waveform_generator_injection = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments_injection)


waveform_generator_injection.parameters = injection_parameters


waveform_generator_recovery = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=memestr.waveforms.phenom.fd_imrx_fast,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator_injection,
                   parameters=injection_parameters)

likelihood_recovery = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_recovery)

result = bilby.core.result.read_in_result(f'outdir_{mode}_injections/run_{run_id}_result.json')


wg_xhm_memory = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_imrx_with_memory)
likelihood_xhm_memory = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wg_xhm_memory)

try:
    log_memory_weights = np.loadtxt(f"{mode}_{run_id}_memory_log_weights")
except Exception:
    reweighted_time_shift_memory_log_bf, log_memory_weights = memestr.postprocessing.reweigh_by_likelihood(
        new_likelihood=likelihood_xhm_memory, result=result,
        reference_likelihood=likelihood_recovery, use_stored_likelihood=True)
    np.savetxt(f"{mode}_{run_id}_memory_log_weights", log_memory_weights)

reweighted_memory_log_bf = logsumexp(log_memory_weights) - np.log(len(log_memory_weights))
n_eff_memory = np.sum(np.exp(log_memory_weights)) ** 2 / np.sum(np.exp(log_memory_weights) ** 2)
print(n_eff_memory)
print(reweighted_memory_log_bf)
