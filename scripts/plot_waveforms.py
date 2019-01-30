import memestr
import bilby
import matplotlib.pyplot as plt
import numpy as np

settings = memestr.core.parameters.AllSettings()
models = [memestr.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
          memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
          memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory]
print(settings.injection_parameters)


def plot_waveform(td_model, mass_ratio):
    settings.waveform_arguments.alpha = 0.1
    settings.waveform_data.duration = 16
    settings.injection_parameters.total_mass = 16
    settings.injection_parameters.mass_ratio = mass_ratio
    waveform_generator = bilby.gw.WaveformGenerator(time_domain_source_model=td_model,
                                                    parameters=settings.injection_parameters.__dict__,
                                                    waveform_arguments=settings.waveform_arguments.__dict__,
                                                    **settings.waveform_data.__dict__)
    plt.plot(waveform_generator.frequency_array, np.abs(waveform_generator.frequency_domain_strain()['plus']))
    plt.title('alpha = ' + str(settings.waveform_arguments.alpha))
    plt.xlim(20, 1000)
    plt.ylim(1e-30, 4e-23)
    plt.loglog()
    # plt.savefig('memory_alpha_' + str(alpha) + '.png')
    plt.show()
    plt.clf()
    # plt.ylim(-0.8e-21, 0.8e-21)
    # plt.plot(waveform_generator.time_array, waveform_generator.time_domain_strain()['plus'], color='black')
    # plt.plot(waveform_generator.time_array, memestr.core.waveforms.tukey(M=len(waveform_generator.time_array), alpha=settings.waveform_arguments.alpha)*1.5e-22, color='blue')
    settings.waveform_arguments.alpha = 0
    waveform_generator = bilby.gw.WaveformGenerator(time_domain_source_model=td_model,
                                                    parameters=settings.injection_parameters.__dict__,
                                                    waveform_arguments=settings.waveform_arguments.__dict__,
                                                    **settings.waveform_data.__dict__)
    plt.plot(waveform_generator.time_array, waveform_generator.time_domain_strain()['plus'], color='red')
    interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
    interferometers.\
        set_strain_data_from_power_spectral_densities(sampling_frequency=settings.waveform_data.sampling_frequency,
                                                      duration=settings.waveform_data.duration,
                                                      start_time=settings.waveform_data.start_time)
    interferometers.inject_signal(parameters=settings.injection_parameters.__dict__,
                                  waveform_generator=waveform_generator)
    plt.show()
    plt.clf()
    network_snr = np.sqrt(np.sum([ifo.meta_data['optimal_SNR']**2 for ifo in interferometers]))
    return network_snr


network_snrs = []
for mass_ratio in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
    network_snrs.append(plot_waveform(memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory, mass_ratio))
plt.show()

print(network_snrs)