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

from setuptools import setup, find_packages

setup(name = 'wdm',
      version = '0.1',
      description = 'Wavelet Directional Method python implementation',
      url = '',
      author = 'Daniel Santiago',
      author_email = 'dspelaez@gmail.com',
      license = 'GNU',
      packages = find_packages(),
      zip_safe = False)
