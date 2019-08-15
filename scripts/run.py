from __future__ import division

import os
import sys
import memestr
import bilby
outdir = sys.argv[1]
dir_path = os.path.dirname(os.path.realpath(__file__))

kwargs = dict()
for arg in sys.argv[2:]:
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

script = memestr.scripts[kwargs['script']]
script(dir_path=dir_path, outdir=outdir, **kwargs)



