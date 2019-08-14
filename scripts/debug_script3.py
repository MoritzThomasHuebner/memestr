import matplotlib

matplotlib.use('Agg')
import memestr.core.waveforms.surrogate

memestr.wrappers.injection_recovery.run_production_injection_imr_phenom(
        # injection_model=frequency_domain_nr_hyb_sur_waveform_with_memory_wrapped,
        recovery_model='frequency_domain_nr_hyb_sur_waveform_without_memory_wrapped',
        filename_base=20021,
        outdir='20021_dynesty_production_IMR_non_mem_rec/', label='IMR_mem_inj_non_mem_rec',
        alpha=0.1, distance_marginalization=False,
        time_marginalization=True, phase_marginalization=True,
        sampler='dynesty', nthreads=1,
        npoints=200, duration=16, random_seed=42, dlogz=0.1,
        sampling_frequency=2048, resume=True, clean=False, n_check_point=100)


# memestr.routines.reweighting.run_reweighting(
#         injection_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory_wrapped,
#         recovery_model=memestr.core.waveforms.time_domain_IMRPhenomD_waveform_with_memory_wrapped,
#         filename_base='20021_dynesty',
#         outdir='20021_dynesty_production_IMR_non_mem_rec/', label='IMR_mem_inj_non_mem_rec',
#         sub_run_id=10,
#         alpha=0.1, distance_marginalization=True,
#         time_marginalization=False, phase_marginalization=False,
#         sampler='dynesty', nthreads=1,
#         npoints=50, duration=16, random_seed=42, dlogz=0.1,
#         sampling_frequency=2048, resume=True, clean=False, n_check_point=100)
