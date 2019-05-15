import bilby
import memestr


memestr.wrappers.injection_recovery. \
    run_production_injection_imr_phenom(
        injection_model=memestr.core.waveforms.time_domain_nr_hyb_sur_waveform_with_memory_wrapped,
        recovery_model=memestr.core.waveforms.frequency_domain_IMRPhenomD_waveform_without_memory,
        filename_base=0,
        outdir='test_production', label='IMR_mem_inj_non_mem_rec',
        alpha=0.1, distance_marginalization=False,
        time_marginalization=False, phase_marginalization=True,
        sampler='dynesty', nthreads=2,
        npoints=400, duration=16, random_seed=42, dlogz=0.1,
        sampling_frequency=2048, resume=True, clean=False, n_check_point=100)

# result = bilby.result.read_in_result('test_production/IMR_mem_inj_non_mem_rec_result.json')
# print(result)
# injected_params = dict(phase=1.4315882007303862, geocent_time=6.92720234665083)
#
# result.plot_corner(parameters=injected_params)
# import deepdish
# ifos = deepdish.io.load('parameter_sets/0_H1L1V1.h5')
#
# ifo = ifos[0]
#
# time_and_phase_shifted_result = memestr.core.postprocessing.\
#     adjust_phase_and_geocent_time_complete_posterior_proper(result=result,
#                                                             ifo=ifo)
# time_and_phase_shifted_result.label = 'time_and_phase_shifted'
# time_and_phase_shifted_result.save_to_file()
# time_and_phase_shifted_result.plot_corner(parameters=injected_params)
#