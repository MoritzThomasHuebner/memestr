import pickle
import sys
import numpy as np
from scipy.special import logsumexp

import bilby
from copy import deepcopy
import memestr
from scipy.optimize import minimize

# event_number = int(sys.argv[1])
event_number = 0
time_tags = ["1126259462-391", "1128678900-4", "1135136350-6", "1167559936-6", "1180922494-5", "1185389807-3",
             "1186302519-7", "1186741861-5", "1187058327-1", "1187529256-5", "1239082262-222168"]
events = ["GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
          "GW170809", "GW170814", "GW170818", "GW170823", "GW190412"]
time_tag = time_tags[event_number]
event = events[event_number]

try:
    result = bilby.core.result.read_in_result('results/{}_precessing/result/run_data0_{}_analysis_H1L1_dynesty_merge_result.json'.format(event, time_tag))
except Exception:
    result = bilby.core.result.read_in_result(
        'results/{}_precessing/result/run_data0_{}_analysis_H1L1V1_dynesty_merge_result.json'.format(event, time_tag))
print(len(result.posterior))
from matplotlib import rc
rc("text", usetex=False)
result.plot_corner(outdir='results')

with open('results/{}/data/run_data0_{}_generation_data_dump.pickle'.format(event, time_tag), "rb") as f:
    data_dump = pickle.load(f)
ifos = data_dump.interferometers


def calculate_overlaps_optimizable(new_params, *args):
    time_shift = new_params[0]
    phase = new_params[1]
    wg_xphm, ref_wave, ifo = args
    wg_xphm.parameters["phase"] = phase
    fd_strain_xphm = wg_xphm.frequency_domain_strain()
    new_wave = memestr.core.waveforms.utils.apply_time_shift_frequency_domain(fd_strain_xphm, ifo.frequency_array, ifo.strain_data.duration, time_shift)
    overlap = memestr.core.postprocessing.overlap_function(a=new_wave, b=ref_wave, frequency=ifo.frequency_array, psd=ifo.power_spectral_density)
    return -overlap

for i in range(1):#len(result.posterior)):
    sample = dict(result.posterior.iloc[i])
    print(sample)
    for k, v in sample.items():
        print(f"{k}: {v}")
    maximum_overlap = 0.
    time_shift = 0.
    new_phase = 0.
    iterations = 0.
    # time_limit = 0.5
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(sample['chirp_mass'], sample['mass_ratio'])
    time_limit = total_mass * 0.000020

    bilby.core.utils.logger.disabled = True
    wg_xphm_meme = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
        frequency_domain_source_model=memestr.core.waveforms.fd_imrxp_22, waveform_arguments=dict(alpha=0.1))
    wg_xphm_meme.parameters = deepcopy(sample)

    wg_xphm_lal = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(waveform_approximant='IMRPhenomXPHM', minimum_frequency=20))
    wg_xphm_lal.parameters = deepcopy(sample)
    bilby.core.utils.logger.disabled = False

    fd_strain_lal = wg_xphm_lal.frequency_domain_strain()
    args = (wg_xphm_meme, fd_strain_lal, ifos[0])

    threshold = 0.99
    res = None
    counter = 0
    while maximum_overlap < threshold and counter < 32:
        if counter == 0:
            init_guess_time = 0.
            init_guess_phase = sample['phase']
            x0 = np.array([init_guess_time, init_guess_phase])
            bounds = [(-time_limit, time_limit), (0, 2*np.pi)]
        else:
            init_guess_time = (2*np.random.random() - 1) * time_limit
            init_guess_phase = np.random.random() * 2*np.pi
            x0 = np.array([init_guess_time, init_guess_phase])
            bounds = [(-time_limit, time_limit), (0, 2*np.pi)]

        res = minimize(calculate_overlaps_optimizable, x0=x0, args=args, bounds=bounds, tol=0.001)
        maximum_overlap = -res.fun
        counter += 1
        print(counter)
    print(i)
    maximum_overlap = -res.fun
    time_shift, new_phase = res.x[0], res.x[1]
    new_phase %= 2 * np.pi
    iterations = res.nit
    print(counter)
    print(maximum_overlap)
    print(time_limit)
    print(time_shift)
    print(sample['phase'])
    print(new_phase)
    print("")
    wg_xphm_meme.parameters['phase'] = new_phase
    import matplotlib.pyplot as plt
    plt.loglog(wg_xphm_lal.frequency_array, np.abs(wg_xphm_lal.frequency_domain_strain()['plus']))
    plt.loglog(wg_xphm_meme.frequency_array, np.abs(wg_xphm_meme.frequency_domain_strain()['plus']
                                                    * np.exp(-2j * np.pi * (ifos[0].strain_data.duration + time_shift) * ifos[0].frequency_array)))
    plt.xlim(20, )
    plt.savefig("test_overlap_abs_plus.png")
    plt.clf()
    plt.loglog(wg_xphm_lal.frequency_array, np.abs(wg_xphm_lal.frequency_domain_strain()['cross']))
    plt.loglog(wg_xphm_meme.frequency_array, np.abs(wg_xphm_meme.frequency_domain_strain()['cross']
                                                    * np.exp(-2j * np.pi * (ifos[0].strain_data.duration + time_shift) * ifos[0].frequency_array)))
    plt.xlim(20, )
    plt.savefig("test_overlap_abs_cross.png")
    plt.clf()

    plt.plot(wg_xphm_lal.frequency_array, np.angle(wg_xphm_lal.frequency_domain_strain()['plus']))
    plt.plot(wg_xphm_meme.frequency_array, np.angle(wg_xphm_meme.frequency_domain_strain()['plus']
                                                      * np.exp(-2j * np.pi * (ifos[0].strain_data.duration + time_shift) * ifos[0].frequency_array)))
    plt.xlim(20, )
    plt.savefig("test_overlap_phase_plus.png")
    plt.clf()
    plt.plot(wg_xphm_lal.frequency_array, np.angle(wg_xphm_lal.frequency_domain_strain()['cross']))
    plt.plot(wg_xphm_meme.frequency_array, np.angle(wg_xphm_meme.frequency_domain_strain()['cross']
                                                    * np.exp(-2j * np.pi * (ifos[0].strain_data.duration + time_shift) * ifos[0].frequency_array)))
    plt.xlim(20, )
    plt.savefig("test_overlap_phase_cross.png")
    plt.clf()

    plt.plot(wg_xphm_lal.time_array, np.real(wg_xphm_lal.time_domain_strain()['plus']))
    plt.plot(wg_xphm_meme.time_array, np.real(wg_xphm_meme.time_domain_strain()['plus']))
    plt.xlim(3.8, )
    plt.savefig("test_overlap_td_plus.png")
    plt.clf()
    plt.plot(wg_xphm_lal.time_array, np.real(wg_xphm_lal.time_domain_strain()['cross']))
    plt.plot(wg_xphm_meme.time_array, np.real(wg_xphm_meme.time_domain_strain()['cross']))
    plt.xlim(3.8, )
    plt.savefig("test_overlap_td_cross.png")
    plt.clf()

# posterior = result.posterior
#
# parameters = dict(posterior.iloc[0])
# wg = bilby.gw.waveform_generator.WaveformGenerator(
#     duration=ifos.duration, sampling_frequency=ifos.sampling_frequency,
#     frequency_domain_source_model=memestr.core.waveforms.phenom.fd_imrx)
# wg_memory = bilby.gw.waveform_generator.WaveformGenerator(
#     duration=ifos.duration, sampling_frequency=ifos.sampling_frequency,
#     frequency_domain_source_model=memestr.core.waveforms.phenom.fd_imrx_with_memory)
#
# likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=wg)
# likelihood_memory = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=wg_memory)
#
# likelihood.parameters = parameters
# likelihood_memory.parameters = parameters



# print(parameters['log_likelihood'])
# print(likelihood.log_likelihood_ratio())
# print(likelihood_memory.log_likelihood_ratio())
#
#
# reweighted_log_bf, log_weights = memestr.core.postprocessing.reweigh_by_likelihood(
#     new_likelihood=likelihood_memory, new_result=result, reference_likelihood=likelihood, reference_result=None)

# np.savetxt("{}_log_weights".format(event), log_weights)





# log_bfs = []
#
# for event in events:
#     log_weights = np.loadtxt("{}_log_weights".format(event))
#     reweighted_log_bf = logsumexp(log_weights) - np.log(len(log_weights))
#     print(reweighted_log_bf)
#     log_bfs.append(reweighted_log_bf)
#
# gwtm_1_original = np.loadtxt("GWTM-1.txt")
#
# import matplotlib
# matplotlib.rcParams.update({'font.size': 13})
# import matplotlib.pyplot as plt
#
# plt.scatter(np.arange(0, 11), log_bfs, label="New (PhenomXHM)")
# plt.scatter(np.arange(0, 10), gwtm_1_original, label="Huebner et al. (NRHybSur)")
# plt.xticks(ticks=np.arange(0, 11), labels=events, rotation=75)
# plt.ylabel("ln BF")
# plt.legend()
# plt.tight_layout()
# plt.savefig("gwtm-1.png")
# plt.clf()
