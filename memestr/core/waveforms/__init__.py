from .phenom import *
from .surrogate import *
from .utils import *
from .ethan import *
from .mwm import *
from .phenom import *

models = dict(
    fd_nr_sur_with_memory=fd_nr_sur_with_memory,
    fd_nr_sur=fd_nr_sur,
    fd_nr_sur_memory_only=fd_nr_sur_memory_only,
    fd_imrd=fd_imrd,
    fd_imrd_with_memory=fd_imrd_with_memory,
    td_imrd_with_memory=td_imrd_with_memory,
    td_imrd=td_imrd,
    td_imrd_memory_only=td_imrd_memory_only
)