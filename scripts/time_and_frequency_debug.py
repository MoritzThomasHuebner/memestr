import bilby as bb
import memestr
from memestr.core.waveforms import *
import matplotlib.pyplot as plt
import gwmemory


settings = memestr.core.parameters.AllSettings()
settings.injection_parameters.total_mass = 20
settings.injection_parameters.mass_ratio = 0.5
waveform_fd_source_model = bb.gw.waveform_generator.WaveformGenerator(
    frequency_domain_source_model=frequency_domain_IMRPhenomD_waveform_without_memory,
    duration=16, sampling_frequency=2048, parameters=settings.injection_parameters.__dict__)

max_index_fd_model = np.argmax(np.abs(np.abs(waveform_fd_source_model.time_domain_strain()['plus'])
                                      + np.abs(waveform_fd_source_model.time_domain_strain()['cross'])))
print(max_index_fd_model)
waveform_td_source_model = bb.gw.waveform_generator.WaveformGenerator(
    time_domain_source_model=time_domain_IMRPhenomD_waveform_without_memory_wrapped,
    duration=16, sampling_frequency=2048, parameters=settings.injection_parameters.__dict__,
    waveform_arguments=dict(max_index_fd_model=max_index_fd_model))

plt.plot(waveform_td_source_model.frequency_array,
         np.abs(waveform_td_source_model.frequency_domain_strain()['plus']),
         label='IMRPhenomD TD model')

plt.plot(waveform_fd_source_model.frequency_array,
         np.abs(waveform_fd_source_model.frequency_domain_strain()['plus']),
         label='IMRPhenomD FD model')

plt.plot(waveform_fd_source_model.frequency_array,
         np.abs(np.abs(waveform_fd_source_model.frequency_domain_strain()['plus']) -
            np.abs(waveform_td_source_model.frequency_domain_strain()['plus'])),
         label='IMRPhenomD residual')


plt.legend()
plt.loglog()
plt.xlabel('f[Hz]')
plt.ylabel('h')
plt.xlim(0, 1024)
plt.show()
plt.clf()


plt.plot(waveform_td_source_model.time_array,
         waveform_td_source_model.time_domain_strain()['plus'],
         label='IMRPhenomD TD model')
plt.plot(waveform_fd_source_model.time_array,
         waveform_fd_source_model.time_domain_strain()['plus'],
         label='IMRPhenomD FD model')

plt.xlabel('t[s]')
plt.ylabel('h')
plt.legend()
plt.show()
plt.clf()


# memestr.wrappers.injection_recovery.\
#     run_basic_injection_imr_phenom(injection_model=memestr.core.waveforms.frequency_domain_IMRPhenomD_waveform_without_memory,
#                                    recovery_model=memestr.core.waveforms.frequency_domain_IMRPhenomD_waveform_without_memory,
#                                    outdir='test', alpha=0.1, zero_noise=True, distance_marginalization=False,
#                                    time_marginalization=False, luminosity_distance=1000.0, nthreads=2, sampler='cpnest',
#                                    dlogz=20, nlive=300, duration=16, random_seed=42, sampling_frequency=2048)