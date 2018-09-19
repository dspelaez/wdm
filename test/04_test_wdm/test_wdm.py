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
    # wavelet coefficients
    omin, omax, nvoice = -6, 1, 32
    frequencies, coefficients = wavelet_spectrogram(A, fs, omin, omax, nvoice, mode='TC98')
    #
    # compute components of wavenumber
    k, l = klcomponents(frequencies, coefficients, x, y, limit=np.pi)
    #
    # compute power density from wavelets coefficients
    power_wavelet = np.abs(coefficients) ** 2 
    #
    # compute directional spectrum and directional spreading
    E_donelan, D = fdir_spectrum(frequencies, dirs, power_wavelet.mean(2), k, l)
    #
    # smooth directional spreading
    D = smooth(D, ws=(30,1))
    #
    # smoth directional spectrum
    E_donelan = smooth(E_donelan, ws=(30,1))
    # ---

    # --- compute direccional and itegrated spectra  ---
    S_jonsnew = interpolate.interp1d(frqs, S_jonswap)(frequencies)
    S_fourier = psd(A, frequencies, fs=fs, winsize=L/4, noverlap=L/8, m=1)
    S_donelan = np.trapz(E_donelan, x=dirs*np.pi/180., axis=0)
    #
    E_jonsnew = S_jonsnew[None,:] * D
    E_fourier = S_fourier[None,:] * D


    # --- plot integrated spectra ---
    fig = plt.figure(figsize=(5, 3.1))
    ax  = fig.add_subplot(111)
    #
    ax.plot([],[], '.w', label="$H_\mathrm{rms} = %.1f\,\mathrm{m}$" % (4.*A.std()))
    spectra_list = {'Jonswap':S_jonsnew, 'Wavelet':S_donelan, 'Fourier':S_fourier}
    for n, s in spectra_list.items():
        computed_hs = 4. * np.trapz(s, frequencies)**.5
        label = "%s: $H_s = %.1f\,\mathrm{m}$" % (n, computed_hs)
        ax.plot(frequencies, s, label=label)
    #
    ax.set_xlabel('$f\,\mathrm{[Hz]}$')
    ax.set_ylabel('$S(f)\,\mathrm{[m^2/Hz]}$')
    ax.legend(loc=0)
    ax.set_xlim([0, .5])
    fig.savefig('integrated_spectra_%s.png' % tag)

    
    # --- plot directional spectra for jonswap ---
    fig = plt.figure(figsize=(4.0,3.1))
    ax = fig.add_subplot(111)
    polar_spectrum(frqs, dirs, E_jonswap, fmax=.4, ax=ax, cbar=True, label=True, smin=-2, smax=2)
    ax.set_title("m$^2$Hz$^{-1}$rad$^{-1}$", loc="right", x=1.25, y=1.025)
    fig.savefig('jonswap_spectra_%s.png' % tag)


    # --- plot directional spectra for each one ---
    fig = plt.figure(figsize=(6.2, 6.2))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    #
    polar_spectrum(frqs, dirs, E_jonswap, fmax=.4, ax=ax1, cbar=False, label=True, smin=-2,smax=2)
    polar_spectrum(frequencies, dirs, E_donelan, fmax=.4, ax=ax2, cbar=False, label=True, smin=-2,
            smax=2)
    polar_spectrum(frequencies, dirs, E_fourier, fmax=.4, ax=ax3, cbar=False, label=True, smin=-2,
            smax=2)
    polar_spectrum(frequencies, dirs, E_jonsnew, fmax=.4, ax=ax4, cbar=False, label=True, smin=-2,
            smax=2)
    #
    ax1.set_title("Jonswap", loc="left", x=0.05, y=1.025)
    ax2.set_title("$E_{ij} = \\overline{|W|^2}_{\\theta=i}$", loc="left", x=0.05, y=1.025)
    ax3.set_title("$E_{ij} = S^F_{j} D_{ij}$",                loc="left", x=0.05, y=1.025)
    ax4.set_title("$E_{ij} = S^J_{j} D_{ij}$",                loc="left", x=0.05, y=1.025)
    #
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax4.set_yticklabels([])
    #
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax4.set_ylabel('')
    #
    fig.savefig('directional_spectra_%s.png' % tag)
# --- }}}


if __name__ == "__main__":

    # --- get an pentagon-shaped array separated 0.9 m ---
    x, y = reg_array(N=5, R=0.9, theta_0=-180)
    co_array(x, y, plot=True, filename='pentagon_coarray.png')

    # --- loop for each case ---
    for tag in ['narrow', 'broad', 'bimodal']:
        main(tag, x, y)
