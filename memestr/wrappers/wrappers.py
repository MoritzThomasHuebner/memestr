import numpy as np
import tupak


def run_basic_injection(injection_model, recovery_model, outdir):
    mass_ratio = 1.5
    total_mass = 60
    S1 = np.array([0, 0, 0])
    S2 = np.array([0, 0, 0])
    s11 = S1[0]
    s12 = S1[1]
    s13 = S1[2]
    s21 = S2[0]
    s22 = S2[1]
    s23 = S2[2]
    LMax = 3
    luminosity_distance = 50.
    inc = np.pi / 2
    phase = 0
    ra = 1.54
    dec = -0.7
    psi = 2.659
    geocent_time = 1126259642.413
    start_time = -0.5
    end_time = 0.01  # 0.029
    duration = end_time - start_time
    sampling_frequency = 2000
    label = 'NRSur'
    tupak.core.utils.setup_logger(outdir=outdir, label=label)
    np.random.seed(88170235)
    injection_parameters = dict(total_mass=total_mass, mass_ratio=mass_ratio, s11=s11, s12=s12, s13=s13, s21=s21,
                                s22=s22, s23=s23, luminosity_distance=luminosity_distance, inc=inc, phase=phase,
                                psi=psi, geocent_time=geocent_time, ra=ra, dec=dec, LMax=LMax)
    waveform_generator = tupak.gw.WaveformGenerator(duration=duration,
                                                    sampling_frequency=sampling_frequency,
                                                    start_time=start_time,
                                                    time_domain_source_model=injection_model,
                                                    parameters=injection_parameters,
                                                    waveform_arguments=dict(LMax=LMax))
    hf_signal = waveform_generator.frequency_domain_strain()
    ifos = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
        name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, duration=duration,
        sampling_frequency=sampling_frequency, start_time=start_time, outdir=outdir) for name in ['H1', 'L1']]
    waveform_generator.time_domain_source_model = recovery_model
    priors = dict()
    for key in ['total_mass', 'mass_ratio', 's11', 's12', 's13', 's21', 's22', 's23', 'luminosity_distance',
                'inc', 'phase', 'ra', 'dec', 'geocent_time', 'psi']:
        priors[key] = injection_parameters[key]
    priors['total_mass'] = tupak.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
    priors['mass_ratio'] = tupak.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$")
    # priors['luminosity_distance'] = tupak.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
    #                                                                      maximum=5e3, latex_label="$L_D$")
    # priors['inc'] = tupak.core.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$")
    priors['phase'] = tupak.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, latex_label="$\phi$")
    # priors['ra'] = tupak.core.prior.Uniform(name='ra', minimum=0, maximum=2*np.pi, latex_label="$RA$")
    # priors['dec'] = tupak.core.prior.Cosine(name='dec', latex_label="$DEC$")
    priors['psi'] = tupak.core.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi, latex_label="$\psi$")
    # priors['geocent_time'] = tupak.core.prior.Uniform(1126259462.322, 1126259462.522, name='geocent_time')
    likelihood = tupak.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                waveform_generator=waveform_generator,
                                                                prior=priors)
    result = tupak.core.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=600,
                                            injection_parameters=injection_parameters, outdir=outdir, label=label)
    result.plot_corner(lionize=True)
    print(result)


def run_basic_injection_imr_phenom(injection_model, recovery_model, outdir):
    mass_ratio = 1.5
    total_mass = 60
    S1 = np.array([0, 0, 0])
    S2 = np.array([0, 0, 0])
    s11 = S1[0]
    s12 = S1[1]
    s13 = S1[2]
    s21 = S2[0]
    s22 = S2[1]
    s23 = S2[2]
    LMax = 3
    luminosity_distance = 500.
    inc = np.pi / 2
    phase = 1.0
    ra = 1.54
    dec = -0.7
    psi = 2.659
    geocent_time = 1126259642.413
    start_time = 0
    end_time = 3
    duration = end_time - start_time
    sampling_frequency = 4096
    label = 'IMRPhenomD'
    tupak.core.utils.setup_logger(outdir=outdir, label=label)
    np.random.seed(88170235)
    injection_parameters = dict(total_mass=total_mass, mass_ratio=mass_ratio, s11=s11, s12=s12, s13=s13, s21=s21,
                                s22=s22, s23=s23, luminosity_distance=luminosity_distance, inc=inc, phase=phase,
                                psi=psi, geocent_time=geocent_time, ra=ra, dec=dec, LMax=LMax)
    waveform_generator = tupak.gw.WaveformGenerator(duration=duration,
                                                    sampling_frequency=sampling_frequency,
                                                    start_time=start_time,
                                                    time_domain_source_model=injection_model,
                                                    parameters=injection_parameters,
                                                    waveform_arguments=dict(LMax=LMax))
    hf_signal = waveform_generator.frequency_domain_strain()
    debug_signal = waveform_generator.time_domain_strain()
    import matplotlib.pyplot as plt
    plt.plot(waveform_generator.time_array, debug_signal['cross'])
    plt.show()
    plt.clf()
    ifos = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
        name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, duration=duration,
        sampling_frequency=sampling_frequency, start_time=start_time, outdir=outdir, zero_noise=True)
        for name in ['H1', 'L1']]
    waveform_generator.time_domain_source_model = recovery_model
    priors = dict()
    for key in ['total_mass', 'mass_ratio', 's11', 's12', 's13', 's21', 's22', 's23', 'luminosity_distance',
                'inc', 'phase', 'ra', 'dec', 'geocent_time', 'psi']:
        priors[key] = injection_parameters[key]
    priors['total_mass'] = tupak.core.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
    # priors['mass_ratio'] = tupak.core.prior.Uniform(minimum=1, maximum=2, latex_label="$q$")
    # priors['luminosity_distance'] = tupak.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e1,
    #                                                                      maximum=5e3, latex_label="$L_D$")
    # priors['inc'] = tupak.core.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$")
    # priors['phase'] = tupak.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, latex_label="$\phi$")
    # priors['ra'] = tupak.core.prior.Uniform(name='ra', minimum=0, maximum=2*np.pi, latex_label="$RA$")
    # priors['dec'] = tupak.core.prior.Cosine(name='dec', latex_label="$DEC$")
    # priors['psi'] = tupak.core.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi, latex_label="$\psi$")
    # priors['geocent_time'] = tupak.core.prior.Uniform(1126259462.322, 1126259462.522, name='geocent_time')
    likelihood = tupak.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                waveform_generator=waveform_generator,
                                                                prior=priors)
    result = tupak.core.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=2000,
                                            injection_parameters=injection_parameters, outdir=outdir, label=label)
    result.plot_corner(lionize=True)
    print(result)