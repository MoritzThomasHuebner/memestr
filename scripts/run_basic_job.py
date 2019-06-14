from __future__ import division
import sys
from memestr.core import submit
from memestr import models, scripts
import os

outdir = sys.argv[1]
script = scripts[sys.argv[2]]
dir_path = os.path.dirname(os.path.realpath(__file__))
recovery_model = models[sys.argv[3]]

kwargs = dict()
for arg in sys.argv[4:]:
    print(arg)
    key = arg.split("=")[0]
    value = arg.split("=")[1]
    if any(char.isdigit() for char in value):
        if all(char.isdigit() for char in value):
            value = int(value)
        else:
            try:
                value = float(value)
            except Exception:
                pass
    kwargs[key] = value
print(kwargs)

submit.run_job(outdir=outdir,
               script=script,
               dir_path=dir_path,
               recovery_model=recovery_model,
               **kwargs)
