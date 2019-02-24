import numpy as np
import bilby as bb
from memestr.core.parameters import AllSettings
from memestr.core.waveforms import time_domain_IMRPhenomD_waveform_with_memory, \
    time_domain_IMRPhenomD_waveform_without_memory

outdir = 'evidence_recalculation/cpnest_example'

result = bb.result.read_in_result(outdir=outdir, label='IMR_mem_inj_non_mem_rec')

settings = AllSettings.from_defaults_with_some_specified_kwargs(duration=16)
settings.injection_parameters.__dict__ = result.injection_parameters

settings.recovery_priors['prior_total_mass'] = bb.core.prior.Uniform(
    minimum=np.maximum(result.injection_parameters.total_mass - 20, 16),
    maximum=result.injection_parameters.total_mass + 20,
    latex_label="$M_{tot}$")
settings.recovery_priors['prior_mass_ratio'] = bb.core.prior.Uniform(
    minimum=np.maximum(result.injection_parameters.mass_ratio - 0.5, 0.4),
    maximum=1,
    latex_label="$q$")
settings.recovery_priors['prior_luminosity_distance'] = bb.gw.prior.UniformComovingVolume(minimum=10,
                                                                                          maximum=2000,
                                                                                          latex_label="$L_D$",
                                                                                          name='luminosity_distance')
settings.recovery_priors['prior_inc'] = bb.core.prior.Uniform(
    minimum=np.maximum(result.injection_parameters.inc - 0.5, 0),
    maximum=np.minimum(result.injection_parameters.inc + 0.5, np.pi),
    latex_label="$\iota$")
settings.recovery_priors['prior_ra'] = bb.core.prior.Uniform(
    minimum=np.maximum(result.injection_parameters.ra - 0.6, 0),
    maximum=np.minimum(result.injection_parameters.ra + 0.6, 2 * np.pi),
    latex_label="$RA$")
settings.recovery_priors['prior_dec'] = bb.core.prior.Uniform(
    minimum=np.maximum(result.injection_parameters.dec - 0.5, -np.pi / 2),
    maximum=np.minimum(result.injection_parameters.dec + 0.5, np.pi / 2),
    latex_label="$DEC$")
settings.recovery_priors['prior_phase'] = bb.core.prior.Uniform(
    minimum=np.maximum(result.injection_parameters.phase - np.pi / 4, 0),
    maximum=np.minimum(result.injection_parameters.phase + np.pi / 4, 2 * np.pi),
    latex_label="$\phi$")
settings.recovery_priors['prior_psi'] = bb.core.prior.Uniform(
    minimum=np.maximum(result.injection_parameters.psi - np.pi / 4, 0),
    maximum=np.minimum(result.injection_parameters.psi + np.pi / 4, 2 * np.pi),
    latex_label="$\psi$")
settings.recovery_priors['prior_geocent_time'] = bb.core.prior.Uniform(
    minimum=result.injection_parameters.geocent_time - 0.1,
    maximum=result.injection_parameters.geocent_time + 0.1,
    latex_label='$t_c$')
waveform_generator = bb.gw.WaveformGenerator(time_domain_source_model=time_domain_IMRPhenomD_waveform_with_memory,
                                             parameters=settings.injection_parameters.__dict__,
                                             waveform_arguments=settings.waveform_arguments.__dict__,
                                             **settings.waveform_data.__dict__)

hf_signal = waveform_generator.frequency_domain_strain()

np.random.seed(31)
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
                                distance_marginalization=False)
likelihood.parameters = settings.injection_parameters.__dict__
print("Injected value")
print(likelihood.log_likelihood())
print(likelihood.log_likelihood_ratio())

df = result.posterior
new_likelihoods = []
for i in range(len(df)):
    for parameter in ['total_mass', 'mass_ratio', 'inc', 'phase',
                      'ra', 'dec', 'psi', 'geocent_time']:
        likelihood.parameters[parameter] = df.iloc[i][parameter]
    df.iloc[i]['log_likelihood'] = likelihood.log_likelihood()
    print(i)

weights = reweighted_result.posterior.log_likelihood - result.posterior.log_likelihood
result.plot_corner(weights=weights, filename='test')
bb.result.plot_multiple([result, reweighted_result], filename='posterior_reweighting/cpnest_example/reweighting_corner')
# print(result)
# print(reweighted_result)
