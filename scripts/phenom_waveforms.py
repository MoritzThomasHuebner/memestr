import matplotlib.pyplot as plt
import memestr.core.waveforms
import tupak
import numpy as np

mass_ratio = 1.5
total_mass = 60
luminosity_distance = 400
s11 = 0
s12 = 0
s13 = 0
s21 = 0
s22 = 0
s23 = 0
inc = np.pi / 2
pol = 0.6
injection_parameters = dict(total_mass=total_mass, mass_ratio=mass_ratio, s11=s11, s12=s12, s13=s13, s21=s21,
                            s22=s22, s23=s23, luminosity_distance=luminosity_distance, inc=inc, pol=pol)

duration = 4
sampling_frequency = 4096
start_time = 0

waveform_generator = tupak.gw.WaveformGenerator(duration=duration,
                                                sampling_frequency=sampling_frequency,
                                                start_time=start_time,
                                                time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
                                                parameters=injection_parameters)
time_domain_data = waveform_generator.time_domain_strain()
frequency_domain_data = waveform_generator.frequency_domain_strain()
fig = plt.figure(figsize=(12, 4))
for key in time_domain_data.keys():
    plt.plot(time_domain_data[key])
plt.show()
plt.close()
