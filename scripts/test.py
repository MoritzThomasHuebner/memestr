from __future__ import division
import numpy as np
import tupak
import matplotlib.pyplot as plt
import core.waveforms as waveforms

mass_ratio = 2
name = 'memester'
total_mass = 60
S1 = np.array([0.8, 0, 0])
S2 = np.array([0, 0.8, 0])
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
ra = 1.375
dec = -1.2108
psi = 2.659
geocent_time = 1126259642.413

starting_time = -0.5
end_time = 0.01  # 0.029

time_duration = end_time - starting_time
sampling_frequency = 2000

outdir = 'outdir'
label = 'test'

tupak.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(88170235)

injection_parameters = dict(total_mass=total_mass, mass_ratio=mass_ratio, s11=s11, s12=s12, s13=s13, s21=s21,
                            s22=s22, s23=s23, luminosity_distance=luminosity_distance, inc=inc, pol=pol,
                            psi=psi, geocent_time=geocent_time, ra=ra, dec=dec, LMax=LMax)

waveform_generator = tupak.WaveformGenerator(time_duration=time_duration,
                                             sampling_frequency=sampling_frequency,
                                             starting_time=starting_time,
                                             time_domain_source_model=waveforms.
                                             time_domain_nr_sur_waveform_without_memory,
                                             parameters=injection_parameters,
                                             waveform_arguments=dict(LMax=LMax))
hf_signal = waveform_generator.frequency_domain_strain()
test_strain_signal = waveform_generator.time_domain_strain()
plt.plot(waveform_generator.time_array, test_strain_signal['plus'])
plt.show()
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, start_time=starting_time, outdir=outdir) for name in ['H1', 'L1']]

priors = dict()
for key in ['total_mass', 'mass_ratio', 's11', 's12', 's13', 's21', 's22', 's23', 'luminosity_distance',
            'inc', 'pol', 'ra', 'dec', 'geocent_time', 'psi']:
    priors[key] = injection_parameters[key]
priors['total_mass'] = tupak.prior.Uniform(minimum=50, maximum=70, latex_label="$M_{tot}$")
priors['luminosity_distance'] = tupak.prior.Uniform(minimum=400, maximum=600, latex_label="$L_{D}$")
priors['inc'] = tupak.prior.Uniform(minimum=0, maximum=np.pi, latex_label="$inc$")

likelihood = tupak.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator,
                                              time_marginalization=False, phase_marginalization=False,
                                              distance_marginalization=False, prior=priors)

result = tupak.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=300,
                           injection_parameters=injection_parameters, outdir=outdir, label=label)

result.plot_corner(lionize=True)
print(result)
