#!/usr/bin/env python

from setuptools import setup

setup(name='memestr',
      version='0.0.5',
      packages=['memestr', 'memestr.waveforms', 'memestr.waveforms.phenom', 'memestr.routines'],
      package_dir={'memestr': 'memestr'},
      )
