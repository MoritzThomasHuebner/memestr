import matplotlib
matplotlib.use('Agg')

import sys

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
memory_log_bfs_injected_degenerate = []
hom_log_bfs = []
hom_log_bfs_injected = []
gw_log_bfs = []
gw_log_bfs_injected = []

min_event_id = int(sys.argv[1])
max_event_id = int(sys.argv[2])

for i in range(min_event_id, max_event_id):
    logger.info(i)
    injection_parameters = get_injection_parameter_set(str(i))
    ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' + str(i) + '_H1L1V1.h5')
    try:
        pp_res = PostprocessingResult.from_json(outdir=str(i)+'_dynesty_production_IMR_non_mem_rec/')
        res = bilby.result.read_in_result(filename=str(i)+'_dynesty_production_IMR_non_mem_rec/IMR_mem_inj_non_mem_rec_result.json')
        gw_log_bf = res.log_bayes_factor
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
    waveform_generator_22 = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=gws_nominal,
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)

    likelihood_memory = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_memory)
    likelihood_no_memory = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_no_memory)
    likelihood_22 = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_22)

    for parameter in ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance',
                      'phase', 'ra', 'dec', 'psi', 'geocent_time', 's13', 's23']:
        likelihood_memory.parameters[parameter] = injection_parameters[parameter]
        likelihood_no_memory.parameters[parameter] = injection_parameters[parameter]
    for parameter in ['luminosity_distance', 'ra', 'dec', 'psi', 'geocent_time']:
        likelihood_22.parameters[parameter] = injection_parameters[parameter]
    mass_1, mass_2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(mass_ratio=injection_parameters['mass_ratio'],
                                                                                       total_mass=injection_parameters['total_mass'])
    likelihood_22.parameters['mass_1'] = mass_1
    likelihood_22.parameters['mass_2'] = mass_2
    likelihood_22.parameters['chi_1'] = injection_parameters['s13']
    likelihood_22.parameters['chi_2'] = injection_parameters['s23']
    likelihood_22.parameters['theta_jn'] = injection_parameters['inc']
    likelihood_22.parameters['phase'] = injection_parameters['phase'] - np.pi/2

    a = likelihood_memory.log_likelihood_ratio()
    b = likelihood_no_memory.log_likelihood_ratio()
    c = likelihood_22.log_likelihood_ratio()
    likelihood_memory.parameters['psi'] += np.pi/2.
    likelihood_memory.parameters['phase'] += np.pi/2.
    likelihood_no_memory.parameters['psi'] += np.pi/2.
    likelihood_no_memory.parameters['phase'] += np.pi/2.
    d = likelihood_memory.log_likelihood_ratio()
    e = likelihood_no_memory.log_likelihood_ratio()

    memory_log_bfs_injected.append(a - b)
    memory_log_bfs_injected_degenerate.append((a - b)/2. + (d - e)/2.)
    memory_log_bfs.append(memory_log_bf)
    hom_log_bfs_injected.append(b - c)
    hom_log_bfs.append(hom_log_bf)
    gw_log_bfs_injected.append(c)
    gw_log_bfs.append(gw_log_bf)
    logger.info("Memory Log BF injected: " + str(memory_log_bfs_injected[-1]))
    logger.info("Memory Log BF injected degenerate: " + str(memory_log_bfs_injected_degenerate[-1]))
    logger.info("Memory Log BF sampled: " + str(memory_log_bfs[-1]))
    logger.info("HOM Log BF injected: " + str(hom_log_bfs_injected[-1]))
    logger.info("HOM Log BF sampled: " + str(hom_log_bfs[-1]))
    logger.info("GW Log BF injected: " + str(gw_log_bfs_injected[-1]))
    logger.info("GW Log BF sampled: " + str(gw_log_bfs[-1]))

# np.random.seed(42)
# np.random.shuffle(memory_log_bfs)
# np.random.seed(42)
# np.random.shuffle(memory_log_bfs_injected)
memory_log_bfs = np.array(memory_log_bfs)
memory_log_bfs_injected = np.array(memory_log_bfs_injected)
memory_log_bfs_injected_degenerate = np.array(memory_log_bfs_injected_degenerate)
memory_log_bfs_cumsum = np.cumsum(memory_log_bfs)
memory_log_bfs_injected_cumsum = np.cumsum(memory_log_bfs_injected)
memory_log_bfs_injected_degenerate_cumsum = np.cumsum(memory_log_bfs_injected_degenerate)
hom_log_bfs = np.array(hom_log_bfs)
hom_log_bfs_injected = np.array(hom_log_bfs_injected)
hom_log_bfs_cumsum = np.cumsum(hom_log_bfs)
hom_log_bfs_injected_cumsum = np.cumsum(hom_log_bfs_injected)
gw_log_bfs = np.array(gw_log_bfs)
gw_log_bfs_injected = np.array(gw_log_bfs_injected)
gw_log_bfs_cumsum = np.cumsum(gw_log_bfs)
gw_log_bfs_injected_cumsum = np.cumsum(gw_log_bfs_injected)
np.savetxt('summary_memory_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt', memory_log_bfs)
np.savetxt('summary_memory_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt', memory_log_bfs_injected)
np.savetxt('summary_memory_log_bfs_injected_degenerate' + str(min_event_id) + '_' + str(max_event_id) + '.txt', memory_log_bfs_injected_degenerate)
np.savetxt('summary_hom_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt', hom_log_bfs)
np.savetxt('summary_hom_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt', hom_log_bfs_injected)
np.savetxt('summary_gw_log_bfs' + str(min_event_id) + '_' + str(max_event_id) + '.txt', gw_log_bfs)
np.savetxt('summary_gw_log_bfs_injected' + str(min_event_id) + '_' + str(max_event_id) + '.txt', gw_log_bfs_injected)
hom_log_bfs = np.array(hom_log_bfs)

# plt.hist(hom_log_bfs, bins=45)
# plt.xlabel('log BFs')
# plt.ylabel('count')
# plt.title('HOM log BFs')
# plt.savefig('summary_hom_hist' + str(min_event_id) + '_' + str(max_event_id))
# plt.clf()

# plt.hist(memory_log_bfs, bins=45)
# plt.xlabel('log BFs')
# plt.ylabel('count')
# plt.title('Memory log BFs')
# plt.tight_layout()
# plt.savefig('summary_memory_hist' + str(min_event_id) + '_' + str(max_event_id))
# plt.clf()

# plt.hist(memory_log_bfs_injected, bins=45)
# plt.xlabel('log BFs')
# plt.ylabel('count')
# plt.title('Memory log BFs injected')
# plt.tight_layout()
# plt.savefig('summary_memory_hist_injected' + str(min_event_id) + '_' + str(max_event_id))
# plt.clf()
#
# plt.plot(memory_log_bfs_injected_cumsum, label='injected', linestyle='--')
# plt.plot(memory_log_bfs_cumsum, label='sampled')
# plt.xlabel('Event ID')
# plt.ylabel('Cumulative log BF')
# plt.legend()
# plt.tight_layout()
# plt.savefig('summary_cumulative_memory_log_bf' + str(min_event_id) + '_' + str(max_event_id))
# plt.clf()
