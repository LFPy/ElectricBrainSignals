#!/usr/bin/env python

from distutils.core import setup
from brainsignals import version


setup(name='BrainSignals',
      version=version.VERSION,
      description='Package for making figures for Electric Brain Signals book',
      author='LFPy-team',
      author_email='lfpy@users.noreply.github.com',
      packages=['brainsignals'],
     )