from __future__ import division
import sys
from memestr.submit import submitter
from memestr import models, scripts
import os

outdir = sys.argv[1]
script = scripts[sys.argv[2]]
dir_path = os.path.dirname(os.path.realpath(__file__))
injection_model = models[sys.argv[3]]
recovery_model = models[sys.argv[4]]

kwargs = dict()
for arg in sys.argv[5:]:
    key = arg.split("=")[0]
    value = arg.split("=")[1]
    if any(char.isdigit() for char in value):
        value = float(value)
    kwargs[key] = value
print(kwargs)

submitter.run_job(outdir=outdir,
                  script=script,
                  dir_path=dir_path,
                  injection_model=injection_model,
                  recovery_model=recovery_model,
                  **kwargs)
