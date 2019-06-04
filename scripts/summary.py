import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from memestr.core.waveforms import *
from memestr.core.parameters import AllSettings
from memestr.core.submit import get_injection_parameter_set
import bilby

settings = AllSettings.from_defaults_with_some_specified_kwargs(alpha=0.1, duration=16, sampling_frequency=2048)

memory_log_bfs = []
memory_log_bfs_injected_bfs = []
hom_log_bfs = []

for i in range(1000, 2000):
    injection_parameters = get_injection_parameter_set(str(i))
    ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(i) + '_H1L1V1.h5')
    try:
        memory_log_bf = np.loadtxt(str(i) + '_pypolychord_production_IMR_non_mem_rec/memory_log_bf.txt')
        hom_log_bf = np.loadtxt(str(i) + '_pypolychord_production_IMR_non_mem_rec/hom_log_bf.txt')
        if memory_log_bf > 1:
            print(i)
        # if memory_log_bf < 1:
        #     memory_log_bfs.append(memory_log_bf)
        # if hom_log_bf < 20:
        #     hom_log_bfs.append(hom_log_bf)
    except OSError as e:
        print(e)
        continue
    waveform_generator_memory = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)
    waveform_generator_no_memory = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)
    likelihood_memory = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_memory)
    likelihood_no_memory = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_no_memory)
    for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                      'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
        likelihood_no_memory.parameters[parameter] = injection_parameters[parameter]
        likelihood_memory.parameters[parameter] = injection_parameters[parameter]
    a = likelihood_memory.log_likelihood_ratio()
    b = likelihood_no_memory.log_likelihood_ratio()
    print(a)
    print(b)
    memory_log_bfs_injected_bfs.append(a - b)

memory_log_bfs = np.array(memory_log_bfs)
memory_log_bfs_injected_bfs = np.array(memory_log_bfs_injected_bfs)
np.random.seed(42)
np.random.shuffle(memory_log_bfs)
np.random.seed(42)
np.random.shuffle(memory_log_bfs_injected_bfs)
memory_log_bfs_cumsum = np.cumsum(memory_log_bfs)
memory_log_bfs_injected_cumsum = np.cumsum(memory_log_bfs_injected_bfs)

hom_log_bfs = np.array(hom_log_bfs)

# plt.hist(hom_log_bfs, bins=30)
# plt.xlabel('log BFs')
# plt.ylabel('count')
# plt.title('HOM log BFs')
# plt.savefig('summary_hom_hist')
# plt.clf()

plt.hist(memory_log_bfs, bins=30)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('Memory log BFs')
plt.savefig('summary_memory_hist')
plt.clf()

plt.hist(memory_log_bfs_injected_bfs, bins=30)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('Memory log BFs injected')
plt.savefig('summary_memory_hist_injected')
plt.clf()


plt.plot(memory_log_bfs_injected_cumsum, label='injected')
plt.plot(memory_log_bfs_cumsum, label='sampled')
plt.xlabel('Event ID')
plt.ylabel('Cummulative log BF')
plt.legend()
plt.savefig('summary_cummulative_memory_log_bf')
plt.clf()

# plt.xlabel('Event ID')
# plt.ylabel('Cummulative log BF')
# plt.savefig('summary_cummulative_memory_log_bf_injected')
# plt.clf()