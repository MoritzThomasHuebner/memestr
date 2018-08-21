from __future__ import division
import sys
from memestr.submit import submitter
from memestr import models, scripts
print(sys.argv)
submitter.run_job(outdir=sys.argv[1],
                  script=scripts[sys.argv[2]],
                  injection_model=models[sys.argv[3]],
                  recovery_model=models[sys.argv[4]])
