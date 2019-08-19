from copy import deepcopy

import bilby
import memestr
import numpy as np

import matplotlib.pyplot as plt

snrs = []
mem_log_bfs_reweight = []
mem_log_bfs_reweight_err = []
mem_log_bfs_injected = []
mem_log_bfs_sampled = []
mem_log_bfs_sampled_err = []

for run_id in range(20000, 20030):
    ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/{}_H1L1V1.h5'.format(run_id))
    snrs.append(np.sqrt(np.sum([ifo.meta_data['matched_filter_SNR'].real**2 for ifo in ifos])))
    mem_log_bfs = []
    no_mem_evidence = []
    mem_evidence = []
    for sub_run_id in range(0, 8):
        pp_res = memestr.core.postprocessing.PostprocessingResult.from_json(outdir='{}_dynesty_production_IMR_non_mem_rec/'.format(run_id),
                                                                            filename='{}pp_result.json'.format(sub_run_id))
        mem_log_bfs.append(pp_res.memory_log_bf)
        res = bilby.result.read_in_result('{}_dynesty_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(run_id, sub_run_id))
        no_mem_evidence.append(res.log_evidence)
    for sub_run_id in range(10, 18):
        res = bilby.result.read_in_result(
            '{}_dynesty_production_IMR_non_mem_rec/{}IMR_mem_inj_non_mem_rec_result.json'.format(run_id, sub_run_id))
        mem_evidence.append(res.log_evidence)
    mem_evidence = np.array(mem_evidence)
    no_mem_evidence = np.array(no_mem_evidence)
    mem_log_bfs_reweight.append(np.mean(mem_log_bfs))
    mem_log_bfs_reweight_err.append(np.std(mem_log_bfs))
    mem_log_bfs_sampled.append(np.mean(mem_evidence - no_mem_evidence))
    mem_log_bfs_sampled_err.append(np.std(mem_evidence - no_mem_evidence))

    injection_parameters = memestr.core.submit.get_injection_parameter_set(run_id)
    settings = memestr.core.parameters.AllSettings.from_defaults_with_some_specified_kwargs(**injection_parameters)
    priors = dict(luminosity_distance=bilby.gw.prior.UniformComovingVolume(minimum=10, maximum=5000,
                                                                           latex_label="$L_D$",
                                                                           name='luminosity_distance'))

    waveform_generator_reweight = bilby.gw.WaveformGenerator(
        time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)

    waveform_generator_recovery = bilby.gw.WaveformGenerator(
        time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
        parameters=deepcopy(settings.injection_parameters.__dict__),
        waveform_arguments=deepcopy(settings.waveform_arguments.__dict__),
        **settings.waveform_data.__dict__)

    likelihood_reweight = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_reweight,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization,
                                    distance_marginalization_lookup_table='.distance_marginalization_lookup.npz')

    likelihood_recovery = bilby.gw.likelihood \
        .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                    waveform_generator=waveform_generator_recovery,
                                    priors=priors,
                                    time_marginalization=settings.other_settings.time_marginalization,
                                    distance_marginalization=settings.other_settings.distance_marginalization,
                                    phase_marginalization=settings.other_settings.phase_marginalization,
                                    distance_marginalization_lookup_table='.distance_marginalization_lookup.npz')
    likelihood_recovery.parameters = injection_parameters
    likelihood_reweight.parameters = injection_parameters
    mem_log_bfs_injected.append(likelihood_reweight.log_likelihood() - likelihood_recovery.log_likelihood())

snrs = np.array(snrs)
mem_log_bfs_reweight = np.array(mem_log_bfs_reweight)

no_mem_evidence = np.array(no_mem_evidence)
mem_evidence = np.array(mem_evidence)
print(snrs)
print(mem_log_bfs_reweight)
print(no_mem_evidence)
print(mem_evidence)
print(mem_evidence - no_mem_evidence)
print(mem_log_bfs_injected)

plt.plot(snrs, mem_log_bfs_injected, label='$\mathcal{L}$ at injected value')
plt.errorbar(snrs, mem_log_bfs_sampled, yerr=mem_log_bfs_sampled_err, label='Sampling', linestyle='None', marker="v")
plt.errorbar(snrs, mem_log_bfs_reweight, yerr=mem_log_bfs_reweight_err, label='Reweighting', linestyle='None', marker="o")
plt.xlabel('$\rho_{\mathrm{mf}}$')
plt.ylabel('$\ln \mathcal{BF}$')
plt.legend()
plt.savefig('snr_vs_evidence.pdf')
# plt.show()
plt.clf()
