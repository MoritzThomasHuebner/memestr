from __future__ import division
import sys
import submitter
from memestr import models, scripts

submitter.run_job(naming_scheme=sys.argv[1],
                  script=scripts[sys.argv[2]],
                  injection_model=models[sys.argv[3]],
                  recovery_model=models[sys.argv[4]])
