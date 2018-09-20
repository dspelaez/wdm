#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     wavestaffs.py
Created:  2017-04-03 17:53
Modified: 2017-04-03 17:53

"""

# --- import libs ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
plt.ion()
#
from wdm.spectra import dirspec, randomphase2d, polar_spectrum
from wdm.coarray import reg_array, co_array
from wdm import wavelet_spectrogram, klcomponents, fdir_spectrum, smooth
#
import os


# --- {{{
def main(tag, x, y):
    """
    This function compares three ways of compute directional specrtum
    using wavelet directional method (donela_eta_al_1996)

    Input:
        tag: case in 'narrow', 'broad', 'bimodal' 
        x,y: position of wavestaffs
    """

    # --- propose a directional spectrum ---
    fs    = 5.0                                                   # <--- sampling frequency
    L     = 2**12                                                 # <--- length of data
    t     = np.arange(0, L/fs, 1./fs)                             # <--- time array
    dt    = t[1] - t[0]                                           # <--- sampling time
    fmin = 2. * fs / L                                            # <--- min frequency
    fmax = fs / 2.                                                # <--- nyquist frequency
    nfrq = int(2 * (fmax - fmin) / fmin)                          # <--- number of frequencies
    frqs = np.linspace(fmin, fmax, nfrq)                          # <--- array of frequencies
    dirs = np.linspace(0, 359, 360)                               # <--- array of directions
    E_broad  = dirspec(frqs, dirs, 2.,  6.,  45., 'cos2s',  1)    # <--- 2d spectrum
    E_narrow = dirspec(frqs, dirs, 2., 14., 225., 'cos2s', 20)    # <--- 2d spectrum

    if tag == 'narrow':
        E_jonswap = E_narrow[:]
        S_jonswap = np.trapz(E_jonswap, x=dirs*np.pi/180., axis=0)
    #
    elif tag == 'broad':
        E_jonswap = E_broad[:]
        S_jonswap = np.trapz(E_jonswap, x=dirs*np.pi/180., axis=0)
    #
    elif tag == 'bimodal':
        E_jonswap = E_broad + E_narrow
        S_jonswap = np.trapz(E_jonswap, x=dirs*np.pi/180., axis=0)
    #
    else:
        raise ValueError('tag must be one of narrow, broad or bimodal')
    # ---


    # --- get time series at x, y points from proposed spectrum ---
    filename = '../data/seasurface.%s' % tag 
    if os.path.exists(filename):
        A = np.genfromtxt(filename)
    else:
        eta = randomphase2d(frqs, dirs, E_jonswap)
        A = np.zeros((len(t), len(x)))
        for i in range(len(x)):
            A[:,i] = eta(t, x[i], y[i])
        #
        np.savetxt(filename, A, fmt='%15.5E')
    # ---

    # --- compute directional spreading from wdm ---
    #
    omin, omax, nvoice = -6, 1, 32
    #
    # compute directional spectrum and directional spreading
    frequencies, directions, E_donelan, D_donelan = fdir_spectrum(
            A, x, y, fs, omin, omax, nvoice, ws=(30,1))

    # --- plot directional spectra for jonswap ---
    fig = plt.figure(figsize=(7.0,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #
    polar_spectrum(frqs, dirs, E_jonswap, fmax=.4,
            ax=ax1, cbar=False, label=True, smin=-2, smax=2)
    ax1.set_title("m$^2$Hz$^{-1}$rad$^{-1}$", loc="right", x=1.25, y=1.025)
    #
    polar_spectrum(frequencies, directions, E_donelan, fmax=.4,
            ax=ax2, cbar=True, label=True, smin=-2, smax=2)
    ax2.set_title("m$^2$Hz$^{-1}$rad$^{-1}$", loc="right", x=1.25, y=1.025)
    fig.savefig('wdm_spectra_%s.png' % tag)

# --- }}}


if __name__ == "__main__":

    # --- get an pentagon-shaped array separated 0.9 m ---
    x, y = reg_array(N=5, R=0.9, theta_0=0)
    co_array(x, y, plot=True, filename='pentagon_coarray.png')

    # --- loop for each case ---
    for tag in ['narrow', 'broad', 'bimodal']:
        main(tag, x, y)
