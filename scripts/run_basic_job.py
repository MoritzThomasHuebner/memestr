from __future__ import division
import sys
from memestr.submit import submitter
from memestr import models, scripts
import os

print(sys.argv)
submitter.run_job(outdir=sys.argv[1],
                  script=scripts[sys.argv[2]],
                  dir_path=os.path.dirname(os.path.realpath(__file__)),
                  injection_model=models[sys.argv[3]],
                  recovery_model=models[sys.argv[4]],
                  luminosity_distance=int(sys.argv[5]))
