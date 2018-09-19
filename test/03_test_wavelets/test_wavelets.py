#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     test_wavelets.py
Created:  2017-04-14 13:40
"""

# --- import libs ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal as signal
import os
#
from wdm import cwt
#
plt.ion()

# --- get time series at x, y points from proposed spectrum ---
for tag in ['broad', 'narrow', 'bimodal', 'harmonic', 'whitenoise']:

    # time array and fmin and fmax
    fs, L = 5., 2**12
    t = np.arange(0, L/fs, 1./fs)
    fmin = 2 * fs / L
    fmax = fs / 2
    #
    # load file of surface elevation
    filename = '../data/seasurface.%s' % tag 
    if os.path.exists(filename):
        x = np.genfromtxt(filename)[:L,0]
    #
    elif tag == 'harmonic':
        f = 2**-4
        x = np.cos(2. * np.pi * f * t) + np.random.randn(L)
    #
    elif tag == 'whitenoise':
        x = np.random.randn(L)
    #
    else:
        raise IOError('Sea surface elevation file not found')
    # ---


    # --- compute wavelet transform and power specrtum ---
    
    # compute Fourier scale of the morlet
    f0 = 6.
    flambda = (4 * np.pi) / (f0 + np.sqrt(2. + f0 ** 2.))

    # wavelet coefficients
    omin, omax, nvoice = -9, 2, 16
    freqs = 2.**np.linspace(omin, omax, nvoice * abs(omin-omax)+1)
    coeffs = cwt(x, fs, freqs)

    # normalized power spectrum
    #   Use this if use Bertran Chapron code
    #    >> delta_freqs = np.gradient(freqs)
    #    >> power = np.abs(coeffs) ** 2 / freqs[:,None]
    # delta_freqs = np.gradient(freqs)
    # power = np.abs(coeffs) ** 2 / freqs[:,None]
    power = np.abs(coeffs) ** 2
    global_power = power.sum(1) / L

    # compute fourier transform
    fft_freqs, fft_power = signal.welch(x, fs=fs, nperseg=L/8, noverlap=L/16, window='hanning')

    # comput cone of influence
    coi = (L/2. - np.abs(np.arange(0, L) - (L-1.)/2.)) * flambda / np.sqrt(2.) / fs

    # reescale power
    Hrms = 4. * x.std()
    m0_fourier = np.trapz(fft_power, x=fft_freqs)
    m0_wavelet = np.trapz(global_power, x=freqs)
    fac = x.var() / m0_wavelet

    # print in screen
    print("-" * 39)
    print(tag)
    print("-" * 39)
    print("Fourier  ---> m0 = {:.2f} - Hs   = {:.2f} m".format(m0_fourier, 4 * m0_fourier**.5))
    print("Wavelets ---> m0 = {:.2f} - Hs   = {:.2f} m".format(m0_wavelet, 4 * m0_wavelet**.5))
    print("Variance ---> m0 = {:.2f} - Hrms = {:.2f} m".format(x.var(), Hrms))
    print("-" * 39)
    print("Factor   ---> {:.8f}".format(fac))
    print("")

    # --- plot power density specrtrums ---
    fig = plt.figure()
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
    ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
    #
    # plot time series
    ax1.plot(t, x, color='k', lw=0.5)
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim([-Hrms, Hrms])
    ax1.set_xticklabels([])
    ax1.set_yticks([-np.floor(Hrms), 0., np.floor(Hrms)])
    ax1.set_ylabel('$\eta \,\mathrm{[m]}$')
    #
    # plot power spectrum
    ax2.contour(t, freqs, power)
    ax2.fill_between(t, 0, 1./coi, color='0.9')
    ax2.set_ylim((fmin, fmax))
    ax2.set_yscale('log', basey=2)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_xlabel('$t\,\mathrm{[s]}$')
    ax2.set_ylabel('$f \,\mathrm{[Hz]}$')
    #
    # plot global spectrum
    ax3.plot(fft_power,    fft_freqs, color='0.75')
    ax3.plot(global_power, freqs,     color='k')
    ax3.set_ylim((fmin, fmax))
    ax3.set_xscale('log', basex=2)
    ax3.set_yscale('log', basey=2)
    ax3.set_yticklabels([])
    ax3.set_xlabel('$|\overline{W}|^2 \,\mathrm{[m^2/Hz]}$')
    #
    fig.savefig('wavelet_%s.png' %  tag)



