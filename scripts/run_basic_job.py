from __future__ import division
import sys
import submitter
from memestr import models, scripts

naming_scheme = sys.argv[1][:-3]
submitter.run_job(naming_scheme=naming_scheme,
                  script=scripts[sys.argv[2]],
                  injection_model=models[sys.argv[3]],
                  recovery_model=models[sys.argv[4]])
