#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# vim:fdm=marker
#
# =============================================================
#  Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     setup.py
Created:  2018-06-13 16:46
"""

from numpy.distutils.core import Extension, setup
from setuptools import find_packages

ext = Extension(name = 'wdm.core',
                sources = ['./wdm/core.f90'])

if __name__ == "__main__":

    setup(name = 'wdm',
          version = '0.2',
          description = 'Python implementation of the Wavelet Directional Method',
          url = 'https://github.com/dspelaez/wdm',
          author = 'Daniel Santiago',
          author_email = 'dspelaez@gmail.com',
          license = 'GNU',
          packages = find_packages(),
          ext_modules = [ext],
          install_requires = [
              "numpy==1.22.0",
              "scipy==1.1.0",
              "matplotlib==2.2.3"
              ],
          zip_safe = False)
