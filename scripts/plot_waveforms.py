import memestr
import bilby
import matplotlib.pyplot as plt
import numpy as np

settings = memestr.submit.parameters.AllSettings()
models = [memestr.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
          memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
          memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory]
print(settings.injection_parameters)

def plot_waveform(td_model):
    for alpha in [0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1]:
        settings.waveform_arguments.alpha = alpha
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

        plt.ylim(-0.8e-21, 0.8e-21)
        plt.plot(waveform_generator.time_array, waveform_generator.time_domain_strain()['plus'])
        plt.show()
        # hf_signal = waveform_generator.frequency_domain_strain()
        # ifo = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
        #     'H1',
        #     injection_polarizations=hf_signal,
        #     injection_parameters=settings.injection_parameters.__dict__,
        #     outdir='test',
        #     zero_noise=False,
        #     **settings.waveform_data.__dict__)
        # plt.plot(ifo.strain_data.frequency_array, np.abs(ifo.strain_data.frequency_domain_strain))
        # plt.title('alpha = ' + str(settings.waveform_arguments.alpha))
        # plt.loglog()
        # plt.show()



plot_waveform(memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory)

# for model in models:
#     plot_waveform(model)
