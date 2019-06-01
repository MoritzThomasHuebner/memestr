import memestr
import memestr.core.waveforms.surrogate

memestr.wrappers.injection_recovery. \
    run_production_injection_imr_phenom(
        injection_model=memestr.core.waveforms.surrogate.time_domain_nr_hyb_sur_waveform_with_memory_wrapped,
        recovery_model=memestr.core.waveforms.frequency_domain_IMRPhenomD_waveform_without_memory,
        filename_base=1007,
        outdir='1007_pypolychord_production_IMR_non_mem_rec', label='IMR_mem_inj_non_mem_rec',
        alpha=0.1, distance_marginalization=True,
        time_marginalization=False, phase_marginalization=True,
        sampler='dynesty', nthreads=1,
        npoints=200, duration=16, random_seed=42, dlogz=0.1,
        sampling_frequency=2048, resume=True, clean=False, n_check_point=100)
