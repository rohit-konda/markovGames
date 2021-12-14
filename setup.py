#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='markovGames',
      version='1.0',
      description='A python package for analyzing markov Games',
      URL='https://github.com/rohit-konda/.git',
      author='Rohit Konda, Bryce Ferguson, Cathy Zhang',
      author_email='rkonda@ucsb.edu',
      license='MIT',
      packages=find_packages(include=["markovGames.*"]),
      install_requires=['numpy'],
      classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
      ],
      zip_safe=False)
