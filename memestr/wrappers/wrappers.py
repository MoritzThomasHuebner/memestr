import numpy as np
import tupak


def run_basic_injection(source_model, outdir):
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
    pol = 0
    ra = 50.
    dec = 30.
    psi = 2.659
    geocent_time = 1126259642.413
    start_time = -0.5
    end_time = 0.01  # 0.029
    duration = end_time - start_time
    sampling_frequency = 2000
    label = 'test'
    tupak.core.utils.setup_logger(outdir=outdir, label=label)
    np.random.seed(88170235)
    injection_parameters = dict(total_mass=total_mass, mass_ratio=mass_ratio, s11=s11, s12=s12, s13=s13, s21=s21,
                                s22=s22, s23=s23, luminosity_distance=luminosity_distance, inc=inc, pol=pol,
                                psi=psi, geocent_time=geocent_time, ra=ra, dec=dec, LMax=LMax)
    waveform_generator = tupak.WaveformGenerator(duration=duration,
                                                 sampling_frequency=sampling_frequency,
                                                 start_time=start_time,
                                                 time_domain_source_model=source_model,
                                                 parameters=injection_parameters,
                                                 waveform_arguments=dict(LMax=LMax))
    hf_signal = waveform_generator.frequency_domain_strain()
    IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
        name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, duration=duration,
        sampling_frequency=sampling_frequency, start_time=start_time, outdir=outdir) for name in ['H1', 'L1']]
    priors = dict()
    for key in ['total_mass', 'mass_ratio', 's11', 's12', 's13', 's21', 's22', 's23', 'luminosity_distance',
                'inc', 'pol', 'ra', 'dec', 'geocent_time', 'psi']:
        priors[key] = injection_parameters[key]
    priors['total_mass'] = tupak.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
    priors['mass_ratio'] = tupak.prior.Uniform(minimum=1, maximum=2, latex_label="$q$")
    # priors['luminosity_distance'] = tupak.prior.UniformComovingVolume(name='luminosity_distance', minimum=1e2, maximum=5e3, latex_label="$L_D$")
    priors['inc'] = tupak.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$\iota$")
    # priors['ra'] = tupak.prior.Uniform(name='ra', minimum=0, maximum=2*np.pi, latex_label="$RA$")
    # priors['dec'] = tupak.prior.Cosine(name='dec', latex_label="$DEC$")
    # priors['psi'] = tupak.prior.Uniform(name='psi', minimum=0, maximum=2 * np.pi, latex_label="$\psi$")
    likelihood = tupak.GravitationalWaveTransient(interferometers=IFOs,
                                                  waveform_generator=waveform_generator,
                                                  prior=priors)
    result = tupak.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=300,
                               injection_parameters=injection_parameters, outdir=outdir, label=label)
    result.plot_corner(lionize=True)
    print(result)