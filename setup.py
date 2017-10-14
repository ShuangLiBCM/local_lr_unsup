#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='local_lr_unsup',
    version='0.0.0',
    description='local learning rule for unsupervised learning',
    author='Shuang Li',
    author_email='shuang.li@bcm.edu',
    url='https://github.com/ShuangLiBCM/local_lr_unsup',
    packages=find_packages(exclude=[]),
    install_requires=['numpy'],
)
