import json
import warnings

from bilby.core.utils import logger
from bilby.core.result import BilbyJsonEncoder

from memestr.core.injection import create_injection

warnings.filterwarnings("ignore")


for i in range(1000):
    logger.info(f'Injection ID: {i}')
    params = create_injection()
    with open(f'testing_injection_creation/{str(i).zfill(3)}.json', 'w') as f:
        out = dict(injections=params)
        json.dump(out, f, indent=2, cls=BilbyJsonEncoder)
