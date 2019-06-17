import time
from memestr.core.submit import get_injection_parameter_set
from memestr.core.parameters import AllSettings
import bilby
import logging
from memestr.core.waveforms import *

priors = dict()
filename_base = '1999'
outdir = 'nr_hyb_sur_rec'
injection_parameters = get_injection_parameter_set(filename_base)
for key in injection_parameters:
    priors['prior_' + key] = injection_parameters[key]
priors['prior_total_mass'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['total_mass'] - 2,
    maximum=injection_parameters['total_mass'] + 2,
    latex_label="$M_{tot}$")
priors['prior_mass_ratio'] = bilby.core.prior.Uniform(
    minimum=0.85,
    maximum=1,
    latex_label="$q$")
priors['prior_luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(minimum=10,
                                                                           maximum=5000,
                                                                           latex_label="$L_D$",
                                                                           name='luminosity_distance')
priors['prior_inc'] = bilby.core.prior.Uniform(minimum=injection_parameters['inc'] - 0.2,
                                               maximum=injection_parameters['inc'] + 0.2,
                                               latex_label="$\\theta_{jn}$")
priors['prior_ra'] = bilby.core.prior.Uniform(minimum=injection_parameters['ra'] - 0.01,
                                              maximum=injection_parameters['ra'] + 0.01,
                                              latex_label="$RA$")
priors['prior_dec'] = bilby.core.prior.Uniform(minimum=injection_parameters['dec'] - 0.01,
                                               maximum=injection_parameters['dec'] + 0.01,
                                               latex_label="$DEC$")
priors['prior_phase'] = bilby.core.prior.Uniform(minimum=0,
                                                 maximum=2 * np.pi,
                                                 latex_label="$\phi$")
priors['prior_psi'] = bilby.core.prior.Uniform(minimum=0,
                                               maximum=np.pi,
                                               latex_label="$\psi$")
priors['prior_geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters['geocent_time'] - 0.1,
                                                        maximum=injection_parameters['geocent_time'] + 0.1,
                                                        latex_label='$t_c$')
priors['prior_s13'] = bilby.gw.prior.AlignedSpin(name='s13', a_prior=bilby.core.prior.Uniform(0.0, 0.5),
                                                 latex_label='s13')
priors['prior_s23'] = bilby.gw.prior.AlignedSpin(name='s23', a_prior=bilby.core.prior.Uniform(0.0, 0.5),
                                                 latex_label='s23')
imr_phenom_kwargs = dict(
    label='IMRPhenomD'
)
logger = logging.getLogger('bilby')
settings = AllSettings.from_defaults_with_some_specified_kwargs(**imr_phenom_kwargs, **injection_parameters)
settings.waveform_data.start_time = settings.injection_parameters.geocent_time + 2 - settings.waveform_data.duration

bilby.core.utils.setup_logger(outdir=outdir, label=settings.sampler_settings.label)
logger.info("Parameter Set: " + str(filename_base))
ifos = bilby.gw.detector.InterferometerList.from_hdf5('parameter_sets/' +
                                                      str(filename_base) +
                                                      '_H1L1V1.h5')
waveform_generator = bilby.gw.WaveformGenerator(frequency_domain_source_model=frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped_no_shift_return,
                                                parameters=settings.injection_parameters.__dict__,
                                                waveform_arguments=settings.waveform_arguments.__dict__,
                                                **settings.waveform_data.__dict__)
priors = deepcopy(settings.recovery_priors.proper_dict())
# likelihood_imr_phenom_unmarginalized = bilby.gw.likelihood \
#     .GravitationalWaveTransient(interferometers=deepcopy(ifos),
#                                 waveform_generator=waveform_generator)
# likelihood_imr_phenom_unmarginalized.parameters = deepcopy(settings.injection_parameters.__dict__)
likelihood_nr_phenom = bilby.gw.likelihood \
    .GravitationalWaveTransient(interferometers=deepcopy(ifos),
                                waveform_generator=waveform_generator,
                                priors=priors,
                                time_marginalization=True,
                                distance_marginalization=True,
                                phase_marginalization=True)
likelihood_nr_phenom.parameters = deepcopy(settings.injection_parameters.__dict__)
np.random.seed(int(time.time() * 1000000) % 1000000)
logger.info('Injection Parameters')
logger.info(str(settings.injection_parameters))

result = bilby.core.sampler.run_sampler(likelihood=likelihood_nr_phenom,
                                        priors=priors,
                                        injection_parameters=deepcopy(settings.injection_parameters.__dict__),
                                        outdir=outdir,
                                        save=True,
                                        verbose=True,
                                        random_seed=np.random.randint(0, 100000),
                                        sampler='dynesty',
                                        npoints=250,
                                        label='NRHybSur',
                                        clean=settings.sampler_settings.clean,
                                        resume=True,
                                        save_bounds=False,
                                        check_point_plot=True,
                                        walks=50,
                                        n_check_point=100)
result.save_to_file()
logger.info(str(result))