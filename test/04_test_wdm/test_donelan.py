#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""
File:     wavestaffs.py
Created:  2017-04-03 17:53
Modified: 2017-04-03 17:53
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
#
import wdm
import wdm.spectra as sp
#
plt.ion()


# test of lake ontario {{{
def donelan_test():
    """Compares data of lake George with the actual implementation."""

    # 
    angle = np.array([0, 33, 105, 177, 249, 321])
    radii = np.array([0, 0.25, 0.25, 0.25, 0.25, 0.25])
    x = radii * np.cos(angle*np.pi/180)
    y = radii * np.sin(angle*np.pi/180)

    # load spectrum
    fsp = sio.loadmat("../data/fsp_82.mat")

    # load surface elevation
    fs = 4
    L = 2**12
    E_list = []
    for n in range(5):
        A = np.genfromtxt("../data/seasurface.lakes")[n*L:(n+1)*L-1,:]

        # compute spectrum
        ws = (10,1)
        limit = np.pi
        omin, omax, nvoice = -5, 1, 64
        frqs, dirs, E0, D = wdm.fdir_spectrum(A, x, y, fs, limit, omin, omax, nvoice, ws)
        E_list.append(E0)

    # plot directional spectra for jonswap
    fig = plt.figure(figsize=(7.0,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #
    sp.polar_spectrum(fsp["f"][0], dirs, fsp["E"]*fsp["f"][0], fmax=1,
            ax=ax1, cbar=False, label=True, smin=-4, smax=-1, thetam=180)
    #
    sp.polar_spectrum(frqs, dirs, np.mean(E_list, 0), fmax=1,
            ax=ax2, cbar=True, label=True, smin=-4, smax=-1)
    ax2.set_title("m$^2$Hz$^{-1}$rad$^{-1}$", loc="right", x=1.25, y=1.025)

    # if not plt.isinteractive():
        # fig.savefig('wdm_spectra_lakes82.png', dpi=600)
# }}}


if __name__ == "__main__":
    donelan_test()
