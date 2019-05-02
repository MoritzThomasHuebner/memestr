import memestr

memestr.wrappers.injection_recovery. \
    run_production_injection_imr_phenom(
        injection_model=memestr.core.waveforms.time_domain_nr_hyb_sur_waveform_with_memory_wrapped,
        recovery_model=memestr.core.waveforms.frequency_domain_IMRPhenomD_waveform_without_memory,
        filename_base=0,
        outdir='test_production', label='IMR_mem_inj_non_mem_rec',
        alpha=0.1, distance_marginalization=False,
        time_marginalization=False, phase_marginalization=False,
        sampler='cpnest', nthreads=2,
        nlive=400, duration=16, random_seed=42, dlogz=0.1,
        sampling_frequency=2048, resume=False)

