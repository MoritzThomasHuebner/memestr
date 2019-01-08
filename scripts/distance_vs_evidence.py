import matplotlib.pyplot as plt
import numpy as np

import bilby
import memestr

mem_inj_mem_rec_data = np.loadtxt(fname="distance_vs_evidence/001_IMR_mem_inj_mem_rec_distance_evidence.dat")
mem_inj_non_mem_rec_data = np.loadtxt(fname="distance_vs_evidence/001_IMR_mem_inj_non_mem_rec_distance_evidence.dat")

mem_inj_mem_rec_data = mem_inj_mem_rec_data[np.argsort(mem_inj_mem_rec_data[:, 0])]
mem_inj_mem_rec_data = mem_inj_mem_rec_data.T
mem_inj_non_mem_rec_data = mem_inj_non_mem_rec_data[np.argsort(mem_inj_non_mem_rec_data[:, 0])]
mem_inj_non_mem_rec_data = mem_inj_non_mem_rec_data.T


@np.vectorize
def memory_bayes_factor(luminosity_distance):
    settings = memestr.core.parameters.AllSettings()
    settings.injection_parameters.luminosity_distance = luminosity_distance
    outdir = "distance_vs_evidence"
    model = memestr.models['time_domain_IMRPhenomD_memory_waveform']
    waveform_generator = bilby.gw.WaveformGenerator(time_domain_source_model=model,
                                                    parameters=settings.injection_parameters.__dict__,
                                                    waveform_arguments=settings.waveform_arguments.__dict__,
                                                    **settings.waveform_data.__dict__)

    hf_signal = waveform_generator.frequency_domain_strain()
    ifos = bilby.gw.detector.InterferometerList(
        [bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
            name,
            injection_polarizations=hf_signal,
            injection_parameters=settings.injection_parameters.__dict__,
            outdir=outdir,
            zero_noise=settings.detector_settings.zero_noise,
            plot=False,
            **settings.waveform_data.__dict__) for name in settings.detector_settings.detectors])
    opt_snr_squared = 0
    for ifo in ifos:
        signal_ifo = ifo.get_detector_response(waveform_polarizations=hf_signal, parameters=settings.injection_parameters.__dict__)
        opt_snr_squared += sum([ifo.optimal_snr_squared(signal_ifo) for ifo in ifos])
    return opt_snr_squared / 2


distances = mem_inj_mem_rec_data[0]
plt.errorbar(distances, mem_inj_mem_rec_data[1] - mem_inj_non_mem_rec_data[1],
             yerr=np.sqrt(np.square(mem_inj_mem_rec_data[3]) + np.square(mem_inj_non_mem_rec_data[3])),
             label='Memory log Bayes Factor')
ideal_bayes_factors = memory_bayes_factor(distances)
plt.plot(distances, ideal_bayes_factors, label='$rho_{opt}^2/2$ for memory waveforms')
plt.plot([0, 1000], [0, 0], '--')
plt.semilogx()
plt.xlabel('Distance in Mpc')
plt.ylabel('log(BF)')
plt.legend()
plt.savefig('distance_vs_evidence/bayes_factors_dynesty_err')
plt.show()
plt.clf()

# Residuals

residuals = (mem_inj_mem_rec_data[1] - mem_inj_non_mem_rec_data[1]) - ideal_bayes_factors
plt.plot(distances, residuals, label='Residuals')
plt.plot([0, 1000], [0, 0], '--')
plt.semilogx()
plt.xlabel('Distance in Mpc')
plt.ylabel('log(BF)')
plt.legend()
plt.savefig('distance_vs_evidence/residuals')
plt.show()
plt.clf()

residual_mean = np.abs(np.mean(residuals))
residual_sigma = np.std(residuals)

plt.errorbar(distances, mem_inj_mem_rec_data[1] - mem_inj_non_mem_rec_data[1],
             yerr=residual_sigma,
             label='Memory log Bayes Factor')
ideal_bayes_factors = memory_bayes_factor(distances)
plt.plot(distances, ideal_bayes_factors, label='$rho_{opt}^2/2$ for memory waveforms')
plt.plot([0, 1000], [0, 0], '--')
plt.semilogx()
plt.xlabel('Distance in Mpc')
plt.ylabel('log(BF)')
plt.legend()
plt.savefig('distance_vs_evidence/bayes_factors_adjusted_err')
plt.show()
plt.clf()

print(residual_mean)
print(residual_sigma)
