from __future__ import division
import sys
from memestr.core import submit
from memestr import scripts
from memestr.core.waveforms import models
import os

outdir = sys.argv[1]
script = scripts[sys.argv[2]]
dir_path = os.path.dirname(os.path.realpath(__file__))

kwargs = dict()
for arg in sys.argv[3:]:
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
               **kwargs)
