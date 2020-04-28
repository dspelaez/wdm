#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
Created:  2017-04-03 17:53
Modified: 2017-04-03 17:53
"""

# --- import libs ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import os
#
import wdm
import wdm.spectra as sp
#
plt.ioff()


# generate spectra {{{
def get_spectra(tag):
    """Perform a test on the Wavelet Directional Method.

    This function compares three ways of compute directional spectrum
    using wavelet directional method (Donelan et al. 1996)

    Args:
        tag (string): case in 'narrow', 'broad', 'bimodal' 
        x,y (array): position of wavestaffs

    Return
        fs, L, time, frqs, dirs, S, E
    """

    # --- propose a directional spectrum ---
    fs, L = 5.0, 2**12                    # <--- samp rate and data len
    time  = np.arange(0, L/fs, 1./fs)     # <--- time array
    frqs  = np.fft.rfftfreq(L, 1/fs)[1:]  # <--- array of frequencies (nonzero)
    dirs  = np.arange(0, 360)             # <--- array of directions
    #
    # create two synthetic spectra, one broad and another narrow
    E_broad  = sp.dirspec(frqs, dirs, 2.,  6.,  45., 'cos2s',  1)
    E_narrow = sp.dirspec(frqs, dirs, 2., 14., 225., 'cos2s', 20)

    if tag == 'narrow':
        E = E_narrow[:]
        S = np.trapz(E, x=dirs*np.pi/180., axis=0)
    #
    elif tag == 'broad':
        E = E_broad[:]
        S = np.trapz(E, x=dirs*np.pi/180., axis=0)
    #
    elif tag == 'bimodal':
        E = E_broad + E_narrow
        S = np.trapz(E, x=dirs*np.pi/180., axis=0)
    #
    else:
        raise ValueError('Arg `tag` must be one of narrow, broad or bimodal')
    # ---

    return fs, L, time, frqs, dirs, S, E
    # }}}

# load sea surface elevation {{{
def get_surface_elevation(tag):
    """Load the surface elevation file in each wavestaff"""

    # --- get time series at x, y points from proposed spectrum ---
    filename = '../data/seasurface.%s' % tag 
    if os.path.exists(filename):
        A = np.genfromtxt(filename)
    else:
        raise IOError("File not found! Check you are in the right path")
    # ---

    return A
    # }}}

# {{{
def main(tag, x, y):
    """Do a comparison of synthetic spectra."""

    # load spectrum and corresponding sea surface elevation
    fs, L, time, frqs_jon, dirs_jon, S_jon, E_jon = get_spectra(tag)
    A = get_surface_elevation(tag)

    # compute directional spectrum and directional spreading
    #
    ws = (30,1)
    # ws = None
    limit = np.pi
    omin, omax, nvoice = -5, 0, 32
    #
    frqs, dirs, E, D = wdm.fdir_spectrum(A, x, y, fs, limit, omin, omax, nvoice, ws)

    # plot directional spectra for jonswap
    fig = plt.figure(figsize=(7.0,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #
    sp.polar_spectrum(frqs_jon, dirs_jon, E_jon, fmax=.4,
            ax=ax1, cbar=False, label=True, smin=-2, smax=2)
    #
    sp.polar_spectrum(frqs, dirs, E, fmax=.4,
            ax=ax2, cbar=True, label=True, smin=-2, smax=2)
    ax2.set_title("m$^2$Hz$^{-1}$rad$^{-1}$", loc="right", x=1.25, y=1.025)

    if not plt.isinteractive():
       fig.savefig('wdm_spectra_%s.png' % tag, dpi=600)

# }}}

if __name__ == "__main__":

    # get an pentagon-shaped array separated 0.9 m
    x, y = wdm.reg_array(N=5, R=0.9, theta_0=0)
    # wdm.co_array(x, y, plot=False)

    # --- loop for each case ---
    for tag in ['narrow', 'broad', 'bimodal']:
        print(tag)
        main(tag, x, y) 

# --- eof ---
