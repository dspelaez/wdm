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
File:     __init__.py
Created:  2018-06-13 17:05
"""

# main module
from .main import *
from .coarray import *

# this module is used only for the tests
from . import spectra
