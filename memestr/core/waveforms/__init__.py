from .phenom import *
from .surrogate import *
from .utils import *
from .ethan import *
from .mwm import *

models = dict(
    frequency_domain_nr_hyb_sur_waveform_with_memory=frequency_domain_nr_hyb_sur_waveform_with_memory,
    frequency_domain_nr_hyb_sur_waveform_without_memory=frequency_domain_nr_hyb_sur_waveform_without_memory,
    frequency_domain_nr_hyb_sur_memory_waveform=frequency_domain_nr_hyb_sur_memory_waveform,
    frequency_domain_IMRPhenomD_waveform_without_memory=frequency_domain_IMRPhenomD_waveform_without_memory,
    frequency_domain_IMRPhenomD_waveform_with_memory=frequency_domain_IMRPhenomD_waveform_with_memory,
    time_domain_IMRPhenomD_waveform_with_memory=time_domain_IMRPhenomD_waveform_with_memory,
    time_domain_IMRPhenomD_waveform_without_memory=time_domain_IMRPhenomD_waveform_without_memory,
    time_domain_IMRPhenomD_memory_waveform=time_domain_IMRPhenomD_memory_waveform
)