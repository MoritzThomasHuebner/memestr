import json
import warnings
from pathlib import Path
import sys

from bilby.core.utils import logger
from bilby.core.result import BilbyJsonEncoder

from memestr.core.injection import create_injection

warnings.filterwarnings("ignore")

if len(sys.argv) > 1:
    minimum_id = int(sys.argv[1])
    maximum_id = int(sys.argv[2])
else:
    minimum_id = 0
    maximum_id = 100

for i in range(minimum_id, maximum_id):
    logger.info(f'Injection ID: {i}')
    params = create_injection()
    Path('injection_parameter_sets').mkdir(parents=True, exist_ok=True)
    with open(f'injection_parameter_sets/{str(i).zfill(3)}.json', 'w') as f:
        out = dict(injections=params)
        json.dump(out, f, indent=2, cls=BilbyJsonEncoder)
