from __future__ import division
import sys
from memestr.submit import submitter
from memestr import models, scripts
import os

submitter.run_job(outdir=sys.argv[1],
                  script=scripts[sys.argv[2]],
                  dir_path=os.path.dirname(os.path.realpath(__file__)),
                  injection_model=models[sys.argv[3]],
                  recovery_model=models[sys.argv[4]])
