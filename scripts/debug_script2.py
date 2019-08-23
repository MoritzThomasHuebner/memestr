import bilby
import memestr
import gwmemory
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import numpy as np

logger = bilby.core.utils.logger


def memory_optimizer(spins, *args):
    params = args[0]
    params['s11'] = spins[0]
    params['s12'] = spins[1]
    params['s13'] = spins[2]
    params['s21'] = spins[3]
    params['s22'] = spins[4]
    params['s23'] = spins[5]
    memory_generator = gwmemory.waveforms.Surrogate(q=parameters['mass_ratio'], MTot=parameters['total_mass'],
                                                    S1=[spins[0], spins[1], spins[2]],
                                                    S2=[spins[3], spins[4], spins[5]],
                                                    distance=parameters['luminosity_distance'])
    memory = memory_generator.time_domain_memory(inc=parameters['inc'], phase=parameters['phase'])
    # wf = bilby.gw.WaveformGenerator(
    #     time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
    #     duration=16, sampling_frequency=4096)
    # max_strain = np.max(np.abs(wf.time_domain_strain(params)['plus']))
    # print(np.max(np.abs(wf.time_domain_strain(params)['cross'])))
    max_strain = np.max(np.abs(memory[0]['plus']))
    print(max_strain)
    return -max_strain*10e22


parameters = memestr.core.parameters.InjectionParameters().__dict__


init_guess_s11 = 0.0001#2*(np.random.random() - 0.5)
init_guess_s12 = 0#2*(np.random.random() - 0.5)
init_guess_s13 = 0#2*(np.random.random() - 0.5)
init_guess_s21 = 0#2*(np.random.random() - 0.5)
init_guess_s22 = 0#2*(np.random.random() - 0.5)
init_guess_s23 = 0#2*(np.random.random() - 0.5)
x0 = np.array([init_guess_s11, init_guess_s12, init_guess_s13,
               init_guess_s21, init_guess_s22, init_guess_s23])
bounds = [(-0.99999, 0.99999), (-0.99999, 0.99999), (-0.99999, 0.99999),
          (-0.99999, 0.99999), (-0.99999, 0.99999), (-0.99999, 0.99999)]

def constraint_fun(xs):
    return np.sqrt(np.sum([x**2 for x in xs]))

# constraint = scipy.optimize.NonlinearConstraint(fun=constraint_fun, lb=0, ub=0.8)
# res = minimize(memory_optimizer, x0=x0, args=parameters, tol=0.00001,
#                constraints=constraint, method='trust-constr')
# print(res)


wf = bilby.gw.WaveformGenerator(
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
    duration=16, sampling_frequency=2048)
wf_no_mem = bilby.gw.WaveformGenerator(
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
    duration=16, sampling_frequency=2048)
parameters['s13'] = 0.99999
parameters['s23'] = 0.99999
plt.plot(wf.time_array, wf_no_mem.time_domain_strain(parameters)['plus'])
plt.plot(wf.time_array, wf.time_domain_strain(parameters)['plus'])
# plt.xlim(2.3, 2.6)
plt.show()
plt.clf()
logger.disabled = True
for dec in np.linspace(0, np.pi, 40):
    parameters['s13'] = 0.9999
    parameters['s23'] = 0.9999
    parameters['mass_ratio'] = 1
    parameters['total_mass'] = 80
    parameters['ra'] = 3.2
    parameters['dec'] = 2.5
    settings = memestr.core.parameters.AllSettings.from_defaults_with_some_specified_kwargs(**parameters)
    settings.detector_settings.zero_noise = True
    hf_signal = wf.frequency_domain_strain(parameters)
    ifos = bilby.gw.detector.InterferometerList([])
    for ifo in ['H1', 'L1', 'V1']:
        logger.disabled = True
        interferometer = memestr.core.population.setup_ifo(hf_signal, ifo, settings)
        logger.disabled = False
        ifos.append(interferometer)

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                waveform_generator=wf)
    likelihood.parameters = parameters

    print(str(likelihood.log_likelihood_ratio()) + str('\t') + str(dec))
