#!/usr/bin/env python

from __future__ import division, print_function

import numpy as np
import bilby

import sys
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
        total_mass=total_mass,
        mass_ratio=mass_ratio,
        s13=0.4,
        s23=0.3,
        luminosity_distance=1000,
        dec=-0.2,
        ra=0.4,
        inc=1.5,
        psi=0.0,
        phase=0.0,
        geocent_time=0.0
    )
    waveform_generator_injection = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=memestr.waveforms.phenom.fd_imrx_fast,
        parameters=injection_parameters)
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
        luminosity_distance=500,
        dec=-0.5,
        ra=2.4,
        theta_jn=1.5,
        psi=0.0,
        phase=0.0,
        geocent_time=0.0
    )
    all_params = bilby.gw.conversion.generate_all_bbh_parameters(sample=injection_parameters)

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
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator_injection,
                   parameters=injection_parameters)

ifos.plot_data(outdir=outdir)

priors = bilby.gw.prior.PriorDict()
priors.from_file(filename=f'aligned_spin_injection.prior')
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 1,
    maximum=injection_parameters['geocent_time'] + 1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_recovery)


result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=200, outdir=outdir, label=label)

result.plot_corner()
