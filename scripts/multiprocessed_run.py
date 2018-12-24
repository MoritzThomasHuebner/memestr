from __future__ import division
import multiprocessing as mp
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
    print(arg)
    key = arg.split("=")[0]
    value = arg.split("=")[1]
    if any(char.isdigit() for char in value):
        if all(char.isdigit() for char in value):
            value = int(value)
        else:
            value = float(value)
    kwargs[key] = value
print(kwargs)


# Define an output queue
output = mp.Queue()

# Setup a list of processes that we want to run
processes = [mp.Process(target=submitter.run_job,
                        kwargs=dict(outdir=outdir,
                                    script=script,
                                    dir_path=dir_path,
                                    injection_model=injection_model,
                                    recovery_model=recovery_model,
                                    **kwargs)) for x in range(2)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]
