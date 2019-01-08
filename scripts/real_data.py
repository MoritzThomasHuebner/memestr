import bilby
import memestr

duration = 8.
sampling_frequency = 4096.

outdir = 'real_data'
label = 'GW150914'
bilby.core.utils.setup_logger(outdir=outdir, label=label)


waveform_generator = bilby.gw.WaveformGenerator(
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory_open_data,
    duration=duration,
    sampling_frequency=sampling_frequency,
    waveform_arguments=dict(alpha=0.1))
interferometers = bilby.gw.detector.get_event_data(label)
interferometers[0].plot_data(signal=interferometers[0].frequency_domain_strain, outdir=outdir)
interferometers[1].plot_data(signal=interferometers[1].frequency_domain_strain, outdir=outdir)
interferometers[0].plot_time_domain_data(outdir=outdir)
interferometers[1].plot_time_domain_data(outdir=outdir)
prior = bilby.gw.prior.BBHPriorDict(filename='GW150914.prior')
prior['a_1'] = bilby.gw.prior.DeltaFunction(0.0)
prior['a_2'] = bilby.gw.prior.DeltaFunction(0.0)
prior['tilt_1'] = bilby.gw.prior.DeltaFunction(0.0)
prior['tilt_2'] = bilby.gw.prior.DeltaFunction(0.0)
prior['phi_12'] = bilby.gw.prior.DeltaFunction(0.0)
prior['phi_jl'] = bilby.gw.prior.DeltaFunction(0.0)
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers, waveform_generator, distance_marginalization=True, priors=prior)

result = bilby.run_sampler(likelihood, prior, sampler='dynesty',
                           outdir=outdir, label=label)
result.plot_corner()
