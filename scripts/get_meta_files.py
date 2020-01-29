import os
from shutil import copyfile


events = ['S190412m', '190503bf', 'S190408an', 'S190421ar', 'S190512at', 'S190513bm', 'S190517h', 'S190519bj',
          'S190521r', 'S190602aq', 'S190630ag', 'S190701ah', 'S190706ai', 'S190707q', 'S190727h', 'S190728q',
          'S190828j', 'S190828l', 'S190915ak', 'S190329w', 'S190910s',
          'S190424ao', 'S190805bq', 'S190413i', 'S190719an']

for event in events:
    copyfile('/home/cbc/public_html/pe/O3/{}/samples/posterior_samples.json'.format(event),
             'metafiles/{}'.format(event))