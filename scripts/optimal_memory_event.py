from __future__ import division, print_function

import numpy as np
import bilby
import memestr
from memestr.core.population import primary_mass_pdf
import logging
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

settings = memestr.core.parameters.AllSettings()

outdir = 'optimal_event'
label = 'optimal_event_snr'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(88170235)


class OptimalSNRLikelihood(bilby.core.likelihood.Likelihood):

    def __init__(self, interferometers, waveform_generator=None):
        super().__init__(dict())
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.waveform_generator.sampling_frequency,
            duration=self.waveform_generator.duration,
            start_time=self.waveform_generator.start_time)

    def log_likelihood(self):
        ifos = copy.deepcopy(self.interferometers)
        logger = logging.getLogger('bilby')
        logger.disabled = True
        ifos.inject_signal(waveform_generator=self.waveform_generator,
                           parameters=self.parameters)
        logger.disabled = False
        return 100*np.sqrt(np.sum([ifo.meta_data['optimal_SNR'] ** 2 for ifo in ifos]))


def find_optimal_snr_from_result():
    res = bilby.core.result.read_in_result(outdir=outdir, label=label)
    settings.injection_parameters.psi = res.posterior.psi.iloc[-1]
    settings.injection_parameters.ra = res.posterior.ra.iloc[-1]
    settings.injection_parameters.dec = res.posterior.dec.iloc[-1]
    wg = bilby.gw.WaveformGenerator(
        duration=settings.waveform_data.duration, sampling_frequency=settings.waveform_data.sampling_frequency,
        time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
        waveform_arguments=settings.waveform_arguments.__dict__, parameters=settings.injection_parameters.__dict__)
    interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=wg.sampling_frequency,
        duration=wg.duration,
        start_time=wg.start_time)
    interferometers.inject_signal(waveform_generator=wg, parameters=settings.injection_parameters.__dict__)
    network_snr = np.sqrt(np.sum([ifo.meta_data['optimal_SNR'] ** 2 for ifo in interferometers]))
    print(network_snr)


wg = bilby.gw.WaveformGenerator(
    duration=settings.waveform_data.duration*4, sampling_frequency=settings.waveform_data.sampling_frequency,
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_memory_waveform,
    waveform_arguments=settings.waveform_arguments.__dict__, parameters=settings.injection_parameters.__dict__)
interferometers = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=wg.sampling_frequency,
    duration=wg.duration,
    start_time=wg.start_time)

# phases = np.linspace(0, 2 * np.pi, 100)
# psis = np.linspace(0, 2 * np.pi, 100)
# phase_old = settings.injection_parameters.phase
# psi_old = settings.injection_parameters.psi
# network_snrs = np.zeros(shape=(100, 100))
# for i, phase in enumerate(phases):
#     for j, psi in enumerate(psis):
#         settings.injection_parameters.psi = psi
#         settings.injection_parameters.phase = phase
#         ifos = copy.deepcopy(interferometers)
#         ifos.inject_signal(waveform_generator=wg, parameters=settings.injection_parameters.__dict__)
#         network_snr = np.sqrt(np.sum([ifo.meta_data['optimal_SNR'] ** 2 for ifo in ifos]))
#         network_snrs[j][i] = network_snr
#
# phase_mesh, psi_mesh = np.meshgrid(phases, psis)
#
# cf = plt.contourf(phase_mesh, psi_mesh, network_snrs)
# plt.xlabel('phase')
# plt.ylabel('psi')
# plt.title('Optimal Network SNR')
# plt.colorbar(cf)
# plt.savefig('opt_snr_vs_phase_psi.png')
# plt.show()
# plt.clf()
#
# settings.injection_parameters.psi = psi_old
# settings.injection_parameters.phase = phase_old


# ras = np.linspace(0, 2 * np.pi, 100)
# decs = np.linspace(-np.pi, np.pi, 100)
# ra_old = settings.injection_parameters.ra
# dec_old = settings.injection_parameters.dec
# network_snrs = np.zeros(shape=(100, 100))
# for i, ra in enumerate(ras):
#     for j, dec in enumerate(decs):
#         settings.injection_parameters.dec = dec
#         settings.injection_parameters.ra = ra
#         ifos = copy.deepcopy(interferometers)
#         ifos.inject_signal(waveform_generator=wg, parameters=settings.injection_parameters.__dict__)
#         network_snr = np.sqrt(np.sum([ifo.meta_data['optimal_SNR'] ** 2 for ifo in ifos]))
#         network_snrs[j][i] = network_snr
#
# ra_mesh, dec_mesh = np.meshgrid(ras, decs)
#
# cf = plt.contourf(ra_mesh, dec_mesh, network_snrs)
# plt.xlabel('ra')
# plt.ylabel('dec')
# plt.title('Optimal Network SNR')
# plt.colorbar(cf)
# plt.savefig('opt_snr_vs_sky_pos.png')
# plt.show()
# plt.clf()
#
# settings.injection_parameters.dec = dec_old
# settings.injection_parameters.ra = ra_old

# 2D mass plot
# tms = np.linspace(20, 200, 100)
# qs = np.linspace(1, 5, 100)
# network_snrs = np.zeros(shape=(100, 100))
# for i, tm in enumerate(tms):
#     for j, q in enumerate(qs):
#         settings.injection_parameters.mass_ratio = q
#         settings.injection_parameters.total_mass = tm
#         ifos = copy.deepcopy(interferometers)
#         ifos.inject_signal(waveform_generator=wg, parameters=settings.injection_parameters.__dict__)
#         network_snr = np.sqrt(np.sum([ifo.meta_data['optimal_SNR'] ** 2 for ifo in ifos]))
#         network_snrs[j][i] = network_snr
# tm_mesh, q_mesh = np.meshgrid(tms, qs)
# cf = plt.contourf(tm_mesh, q_mesh, network_snrs)
# plt.xlabel('total_mass')
# plt.ylabel('mass ratio')
# plt.title('Optimal Network SNR')
# plt.colorbar(cf)
# plt.savefig('opt_snr_vs_mass_params.png')
# plt.show()
# plt.clf()

pm, pm_pdf = primary_mass_pdf(1000)
pm[-1] = 150
pm_pdf[-1] = 0
# tms = np.linspace(10, 300, 100)
# network_snrs = np.zeros(100)
# for i, tm in enumerate(tms):
#     settings.injection_parameters.mass_ratio = 1
#     settings.injection_parameters.total_mass = tm
#     ifos = copy.deepcopy(interferometers)
#     ifos.inject_signal(waveform_generator=wg, parameters=settings.injection_parameters.__dict__)
#     network_snr = np.sqrt(np.sum([ifo.meta_data['optimal_SNR'] ** 2 for ifo in ifos]))
#     network_snrs[i] = network_snr
#
# fig, ax1 = plt.subplots()
# ax1.plot(tms, network_snrs, color='b')
# ax1.set_xlabel('total_mass')
# ax1.set_ylabel('optimal SNR', color='b')
# ax1.tick_params('y', colors='b')
# ax2 = ax1.twinx()
# ax2.plot(2 * pm, pm_pdf, color='r')
# ax2.set_ylabel('Total mass pdf', color='r')
# ax2.tick_params('y', colors='r')
# plt.savefig('optimal_event/opt_snr_vs_total_mass.png')
# plt.show()
# plt.clf()

wg_no_mem = bilby.gw.WaveformGenerator(
    duration=settings.waveform_data.duration*4, sampling_frequency=settings.waveform_data.sampling_frequency,
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_without_memory,
    waveform_arguments=settings.waveform_arguments.__dict__, parameters=settings.injection_parameters.__dict__)
wg_with_mem = bilby.gw.WaveformGenerator(
    duration=settings.waveform_data.duration*4, sampling_frequency=settings.waveform_data.sampling_frequency,
    time_domain_source_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory,
    waveform_arguments=settings.waveform_arguments.__dict__, parameters=settings.injection_parameters.__dict__)

tms = np.linspace(10, 300, 100)
log_evidences = []
settings.injection_parameters.luminosity_distance = 1000
for i, tm in enumerate(tms):
    settings.injection_parameters.mass_ratio = 1
    settings.injection_parameters.total_mass = tm
    ifos = bilby.gw.detector.InterferometerList(
        [bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
            name,
            injection_polarizations=wg_with_mem.frequency_domain_strain(settings.injection_parameters.__dict__),
            injection_parameters=settings.injection_parameters.__dict__,
            outdir=outdir,
            zero_noise=settings.detector_settings.zero_noise,
            plot=False,
            duration=settings.waveform_data.duration*4,
            sampling_frequency=settings.waveform_data.sampling_frequency) for name in settings.detector_settings.detectors])
    likelihood1 = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                 waveform_generator=wg_with_mem)
    likelihood2 = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                 waveform_generator=wg_no_mem)
    likelihood1.parameters = settings.injection_parameters.__dict__
    likelihood2.parameters = settings.injection_parameters.__dict__
    log_evidence = likelihood1.log_likelihood() - likelihood2.log_likelihood()
    log_evidences.append(log_evidence)


fig, ax1 = plt.subplots()
ax1.plot(tms, log_evidences, color='b')
ax1.set_xlabel('total_mass')
ax1.set_ylabel('log evidence', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(2 * pm, pm_pdf, color='r')
ax2.set_ylabel('Total mass pdf', color='r')
ax2.tick_params('y', colors='r')
plt.savefig('optimal_event/log_evidence_vs_total_mass.png')
plt.show()
plt.clf()

log_evidence_func = interp1d(tms, log_evidences)
tm_pdf_func = interp1d(2 * pm, pm_pdf)

plt.plot(2*pm, log_evidence_func(2*pm) * tm_pdf_func(2*pm))
plt.xlabel('Total mass')
plt.ylabel('log evidence times total mass pdf')
plt.savefig('optimal_event/true_log_bf_dist.png')
plt.show()
plt.clf()
# priors = settings.recovery_priors.proper_dict()
# for key in ['phase', 'luminosity_distance', 'geocent_time', 'total_mass', 'mass_ratio',
#             's11', 's12', 's13', 's21', 's22', 's23']:
#     priors[key] = settings.injection_parameters.__dict__[key]
#
#
#
# likelihood = OptimalSNRLikelihood(
#     interferometers=bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1']),
#     waveform_generator=wg)
#
# result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500, outdir=outdir,
#                            label=label, sample='unif')
#
# result.plot_corner()
