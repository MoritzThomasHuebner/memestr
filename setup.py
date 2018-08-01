#!/usr/bin/env python

from distutils.core import setup
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
setup(name='memestr',
      version='0.0.1',
      packages=['memestr', 'memestr.core', 'memestr.wrappers'],
      package_dir={'memestr': 'memestr'},
      )
