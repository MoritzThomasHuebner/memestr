from copy import deepcopy

import bilby
import memestr
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# snrs = np.loadtxt('SNR_VS_LOGBF_DATA/snrs.txt')
# mem_log_bfs_reweight = []
# mem_log_bfs_reweight_err = []
# mem_log_bfs_injected = []
# mem_log_bfs_sampled = []
# mem_log_bfs_sampled_err = []
#
# for run_id in range(32, 48):
#     run_id = str(run_id).zfill(3)
#     with open('SNR_VS_LOGBF_DATA/{}_reweighing_result/0_7.json'.format(run_id)) as f:
#         data = json.load(f)
#
#     injection_bf = data['injection_bfs']['0']
#     sampling_bfs = [data['sampling_bfs'][str(i)] for i in range(8)]
#     reweighing_to_memory_bfs = [data['reweighing_to_memory_bfs_mem_inj'][str(i)] for i in range(8)]
#
#     mem_log_bfs_reweight.append(np.mean(reweighing_to_memory_bfs))
#     mem_log_bfs_reweight_err.append(np.std(reweighing_to_memory_bfs) / np.sqrt(len(reweighing_to_memory_bfs)))
#     mem_log_bfs_sampled.append(np.mean(sampling_bfs))
#     mem_log_bfs_sampled_err.append(np.std(sampling_bfs) / np.sqrt(len(sampling_bfs)))
#     mem_log_bfs_injected.append(injection_bf)

mem_log_bfs_reweight = []
mem_log_bfs_reweight_err = []
mem_log_bfs_sampled = []
mem_log_bfs_sampled_err = []
mem_log_bfs_injected = []
snrs = []

no_mem_model = memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory
mem_model = memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory
settings = memestr.core.parameters.AllSettings()
waveform_generator = bilby.gw.WaveformGenerator(time_domain_source_model=no_mem_model,
                                                parameters=settings.injection_parameters.__dict__,
                                                waveform_arguments=settings.waveform_arguments.__dict__,
                                                **settings.waveform_data.__dict__)
# ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/20000_H1L1V1.h5')
# priors = deepcopy(settings.recovery_priors.proper_dict())
# likelihood = bilby.gw.likelihood \
#     .GravitationalWaveTransient(interferometers=deepcopy(ifos),
#                                 waveform_generator=waveform_generator,
#                                 priors=priors,
#                                 distance_marginalization=True)

for run_id in range(20000, 20030):
    res_mem_rec = bilby.core.result.read_in_result(
        '{}_dynesty_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(run_id, 10))
    snrs.append(np.sqrt(np.sum([res_mem_rec.meta_data['likelihood']['interferometers'][ifo]['optimal_SNR']**2 for ifo in ['H1', 'L1', 'V1']])))
    continue

    # sampling_log_bfs = []
    # reweight_log_bfs = []
    # for i in range(0, 8):
    #     res_non_mem_rec = bilby.core.result.read_in_result('{}_dynesty_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(run_id, i))
    #     res_mem_rec = bilby.core.result.read_in_result('{}_dynesty_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(run_id, i + 10))
    #     pp_result = memestr.core.postprocessing.PostprocessingResult.from_json('{}_dynesty_production_IMR_non_mem_rec/'.format(run_id), '{}pp_result.json'.format(i))
    #     sampling_log_bfs.append(res_mem_rec.log_bayes_factor - res_non_mem_rec.log_bayes_factor)
    #     reweight_log_bfs.append(pp_result.memory_log_bf)
    # mem_log_bfs_reweight.append(np.mean(reweight_log_bfs))
    # mem_log_bfs_reweight_err.append(np.std(reweight_log_bfs)/np.sqrt(len(reweight_log_bfs)))
    # mem_log_bfs_sampled.append(np.mean(sampling_log_bfs))
    # mem_log_bfs_sampled_err.append(np.std(sampling_log_bfs)/np.sqrt(len(sampling_log_bfs)))
    #
    params = memestr.core.submit.get_injection_parameter_set(run_id)
    del params['luminosity_distance']
    likelihood.interferometers = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/{}_H1L1V1.h5'.format(run_id))
    likelihood.parameters = memestr.core.submit.get_injection_parameter_set(run_id)
    likelihood.waveform_generator.time_domain_source_model = mem_model
    mem_evidence = likelihood.log_likelihood_ratio()
    print("memory evidence: " + str(mem_evidence))
    likelihood.waveform_generator.time_domain_source_model = no_mem_model
    likelihood.waveform_generator._cache['model'] = 'test'
    no_mem_evidence = likelihood.log_likelihood_ratio()
    print("no memory evidence: " + str(no_mem_evidence))
    mem_log_bfs_injected.append(mem_evidence - no_mem_evidence)

    print(run_id)
    print(mem_log_bfs_injected[-1])
    # print(mem_log_bfs_reweight[-1])
    # print(mem_log_bfs_sampled[-1])
    # print(mem_log_bfs_sampled_err[-1])

res = np.loadtxt('SNR_VS_LOGBF_DATA/new_data.txt')
mem_log_bfs_reweight = res[0]
mem_log_bfs_reweight_err = res[1]
mem_log_bfs_sampled = res[2]
mem_log_bfs_sampled_err = res[3]
np.savetxt('SNR_VS_LOGBF_DATA/new_data.txt', np.array([snrs, mem_log_bfs_reweight, mem_log_bfs_reweight_err,
                                                       mem_log_bfs_sampled, mem_log_bfs_sampled_err]))

fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.plot(snrs, mem_log_bfs_sampled, label='Sampling', linestyle='None', marker="v")
ax0.plot(snrs, mem_log_bfs_reweight, label='Reweighting', linestyle='None', marker="o")
ax0.plot(snrs, mem_log_bfs_injected, label='$\ln\mathcal{L}$ at injected value')
ax0.set_ylabel('$\ln \mathcal{BF}_{\mathrm{mem}}$')
ax0.legend()
ax0.set_xticks([])


ax1.plot(snrs, mem_log_bfs_sampled_err, label='Sampling', linestyle='None', marker="v")
ax1.plot(snrs, mem_log_bfs_reweight_err, label='Reweighting', linestyle='None', marker="o")
ax1.set_yscale('log')
ax1.set_xlabel('$\\rho_{mf}$')
# ax1.set_ylim(1e-4, 1)
# ax1.set_yticks([10e-4, 10e-1])
ax1.set_xticks([10, 20, 30, 40, 50, 60])
ax1.set_ylabel('$\Delta \ln \mathcal{BF}_{\mathrm{mem}}$')
plt.tight_layout()
plt.savefig('snr_vs_evidence_new_data.pdf')
plt.show()
plt.clf()
