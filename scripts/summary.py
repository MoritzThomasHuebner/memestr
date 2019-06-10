import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from memestr.core.waveforms import *
from memestr.core.parameters import AllSettings
from memestr.core.postprocessing import PostprocessingResult
from memestr.core.submit import get_injection_parameter_set
import bilby

logger = bilby.core.utils.logger

logger.disabled = False

settings = AllSettings.from_defaults_with_some_specified_kwargs(alpha=0.1, duration=16, sampling_frequency=2048)

memory_log_bfs = []
memory_log_bfs_injected = []
hom_log_bfs = []

for i in range(1900, 2000):
    logger.info(i)
    injection_parameters = get_injection_parameter_set(str(i))
    ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(i) + '_H1L1V1.h5')
    try:
        pp_res = PostprocessingResult.from_json(outdir=str(i)+'_dynesty_production_IMR_non_mem_rec/')
        memory_log_bf = pp_res.memory_log_bf
        hom_log_bf = pp_res.hom_log_bf
        if memory_log_bf is None:
            continue
        # if memory_log_bf > 1:
        #     logger.info(i)
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
    memory_log_bfs_injected.append(a - b)
    memory_log_bfs.append(memory_log_bf)
    hom_log_bfs.append(hom_log_bf)
    logger.info(memory_log_bfs_injected[-1])
    logger.info(memory_log_bfs[-1])

memory_log_bfs = np.array(memory_log_bfs)
memory_log_bfs_injected = np.array(memory_log_bfs_injected)
np.random.seed(42)
np.random.shuffle(memory_log_bfs)
np.random.seed(42)
np.random.shuffle(memory_log_bfs_injected)
memory_log_bfs_cumsum = np.cumsum(memory_log_bfs)
memory_log_bfs_injected_cumsum = np.cumsum(memory_log_bfs_injected)
np.savetxt('summary_log_bfs.txt', memory_log_bfs)
np.savetxt('summary_log_bfs_injected.txt', memory_log_bfs_injected)
hom_log_bfs = np.array(hom_log_bfs)

plt.hist(hom_log_bfs, bins=45)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('HOM log BFs')
plt.savefig('summary_hom_hist')
plt.clf()
#
plt.hist(memory_log_bfs, bins=45)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('Memory log BFs')
plt.tight_layout()
plt.savefig('summary_memory_hist')
plt.clf()
#
plt.hist(memory_log_bfs_injected, bins=45)
plt.xlabel('log BFs')
plt.ylabel('count')
plt.title('Memory log BFs injected')
plt.tight_layout()
plt.savefig('summary_memory_hist_injected')
plt.clf()

plt.plot(memory_log_bfs_injected_cumsum, label='injected', linestyle='--')
plt.plot(memory_log_bfs_cumsum, label='sampled')
plt.xlabel('Event ID')
plt.ylabel('Cummulative log BF')
plt.legend()
plt.tight_layout()
plt.savefig('summary_cummulative_memory_log_bf')
plt.clf()

# plt.xlabel('Event ID')
# plt.ylabel('Cummulative log BF')
# plt.savefig('summary_cummulative_memory_log_bf_injected')
# plt.clf()
