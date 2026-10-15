#!/usr/bin/env python

from setuptools import setup, find_packages
from brainsignals import version

setup(
    name='BrainSignals',
    version=version.VERSION,
    description='Package for making figures for Electric Brain Signals book',
    author='LFPy-team',
    author_email='lfpy@users.noreply.github.com',
    packages=find_packages(),
)
