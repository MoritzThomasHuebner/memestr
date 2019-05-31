#!/usr/bin/env python

from distutils.core import setup

setup(name='memestr',
      version='0.0.2',
      packages=['memestr', 'memestr.core', 'memestr.core.waveforms', 'memestr.wrappers'],
      package_dir={'memestr': 'memestr'},
      )
