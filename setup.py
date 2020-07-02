#!/usr/bin/env python

from setuptools import setup

setup(name='memestr',
      version='0.0.4',
      packages=['memestr', 'memestr.core', 'memestr.core.waveforms', 'memestr.core.waveforms.phenom', 'memestr.routines'],
      package_dir={'memestr': 'memestr'},
      )
