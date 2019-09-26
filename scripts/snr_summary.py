from copy import deepcopy

import bilby
import memestr
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

snrs = np.loadtxt('SNR_VS_LOGBF_DATA/snrs.txt')
mem_log_bfs_reweight = []
mem_log_bfs_reweight_err = []
mem_log_bfs_injected = []
mem_log_bfs_sampled = []
mem_log_bfs_sampled_err = []

for run_id in range(32, 48):
    run_id = str(run_id).zfill(3)
    with open('SNR_VS_LOGBF_DATA/{}_reweighing_result/0_7.json'.format(run_id)) as f:
        data = json.load(f)
    # print(data)

    injection_bf = data['injection_bfs']['0']
    sampling_bfs = [data['sampling_bfs'][str(i)] for i in range(8)]
    reweighing_to_memory_bfs = [data['reweighing_to_memory_bfs_mem_inj'][str(i)] for i in range(8)]

    mem_log_bfs_reweight.append(np.mean(reweighing_to_memory_bfs))
    mem_log_bfs_reweight_err.append(np.std(reweighing_to_memory_bfs) / np.sqrt(len(reweighing_to_memory_bfs)))
    mem_log_bfs_sampled.append(np.mean(sampling_bfs))
    mem_log_bfs_sampled_err.append(np.std(sampling_bfs) / np.sqrt(len(sampling_bfs)))
    mem_log_bfs_injected.append(injection_bf)
#     no_mem_evidence = np.array(no_mem_evidence)
#     mem_log_bfs_reweight.append(np.mean(mem_log_bfs))
#     mem_log_bfs_reweight_err.append(np.std(mem_log_bfs)/np.sqrt(len(mem_log_bfs)))
#     mem_log_bfs_sampled.append(np.mean(mem_evidence - no_mem_evidence))
#     mem_log_bfs_sampled_err.append(np.std(mem_evidence - no_mem_evidence)/np.sqrt(len(mem_evidence)))
#
#     injection_parameters = memestr.core.submit.get_injection_parameter_set(run_id)
#     settings = memestr.core.parameters.AllSettings.from_defaults_with_some_specified_kwargs(**injection_parameters)
#     priors = dict(luminosity_distance=bilby.gw.prior.UniformComovingVolume(minimum=10, maximum=5000,
#                                                                            latex_label="$L_D$",
#                                                                            name='luminosity_distance'))
#
#     waveform_generator_reweight = bilby.gw.WaveformGenerator(
#         time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
#         parameters=deepcopy(settings.injection_parameters.__dict__),
#         waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
#         **settings.waveform_data.__dict__)
#
#     waveform_generator_recovery = bilby.gw.WaveformGenerator(
#         time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
#         parameters=deepcopy(settings.injection_parameters.__dict__),
#         waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
#         **settings.waveform_data.__dict__)
#
#     likelihood_reweight = bilby.gw.likelihood \
#         .GravitationalWaveTransient(interferometers=deepcopy(ifos),
#                                     waveform_generator=waveform_generator_reweight,
#                                     priors=priors,
#                                     time_marginalization=settings.other_settings.time_marginalization,
#                                     distance_marginalization=settings.other_settings.distance_marginalization,
#                                     phase_marginalization=settings.other_settings.phase_marginalization,
#                                     distance_marginalization_lookup_table='.distance_marginalization_lookup.npz')
#
#     likelihood_recovery = bilby.gw.likelihood \
#         .GravitationalWaveTransient(interferometers=deepcopy(ifos),
#                                     waveform_generator=waveform_generator_recovery,
#                                     priors=priors,
#                                     time_marginalization=settings.other_settings.time_marginalization,
#                                     distance_marginalization=settings.other_settings.distance_marginalization,
#                                     phase_marginalization=settings.other_settings.phase_marginalization,
#                                     distance_marginalization_lookup_table='.distance_marginalization_lookup.npz')
#     likelihood_recovery.parameters = injection_parameters
#     likelihood_reweight.parameters = injection_parameters
#     mem_log_bfs_injected.append(likelihood_reweight.log_likelihood() - likelihood_recovery.log_likelihood())
#
# snrs = np.array(snrs)
# mem_log_bfs_reweight = np.array(mem_log_bfs_reweight)
#
# no_mem_evidence = np.array(no_mem_evidence)
# mem_evidence = np.array(mem_evidence)
# print(snrs)
# print(mem_log_bfs_reweight)
# print(mem_log_bfs_injected)
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

# axs[0].xlabel('$\mathrm{SNR}_{\mathrm{mf}}$')
# axs[0].ylabel('$\ln \mathcal{BF}$')


ax1.plot(snrs, mem_log_bfs_sampled_err, label='Sampling', linestyle='None', marker="v")
ax1.plot(snrs, mem_log_bfs_reweight_err, label='Reweighting', linestyle='None', marker="o")
ax1.set_yscale('log')
ax1.set_xlabel('SNR')
ax1.set_ylim(1e-4, 1)
# ax1.set_yticks([10e-4, 10e-1])
ax1.set_xticks([10, 20, 30, 40, 50, 60])
ax1.set_ylabel('$\Delta \ln \mathcal{BF}$')
plt.tight_layout()
plt.savefig('snr_vs_evidence.pdf')
plt.show()
plt.clf()
