#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     1d_spectrum.py
Created:  2017-03-18 15:32
Modified: 2017-03-18 15:32

Purpose:
    This scripts uses the libraries to compute the well-known shape
    of a JONSWAP spectrum and synthesize the surface elevation by
    using the random phase method.
"""

# --- import libs ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
#
from tools.wdm import cwt
from tools.spectra import pwelch, jonswap, randomphase1d
#
plt.ion()


# === example of jonswap === {{{
# create axes
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111)

# loop for several periods
f = np.linspace(0.001, 1., 1000)
for i in [2,4,8,12]:
    S = jonswap(f, 1, i)
    ax.plot(f, S, lw=1.5, label='$T_p = %d\,\mathrm{s}$' % i)

# add legend and labels
ax.legend(loc=0, frameon=False)
ax.set_xlabel('$f\,\mathrm{[Hz]}$')
ax.set_ylabel('$S\,(f)\,\mathrm{[m^2/Hz]}$')
ax.set_ylim([0, 2.5])

# savefigure
fig.savefig('jonswap_scalar.png')
# }}}


# === synthesizing the spectrum === {{{
# --- define parameters ---
fs   = 5.0                                  # <--- sampling frequency
L    = 2**12                                # <--- length of data
t    = np.arange(0, L/fs, 1./fs)            # <--- time array of 13.6 mins
fmin = 1. / (0.5 * len(t) / fs)             # <--- min frequency
fmax = fs / 2.                              # <--- nyquist frequency
nfrq = int(2 * (fmax - fmin) / fmin)        # <--- number of frequencies
frqs = np.linspace(fmin, fmax, nfrq)        # <--- array of frequencies
Sbroad  = jonswap(frqs, 2.,  6., 3.3)       # <--- compute jonswap spectrum
Snarrow = jonswap(frqs, 2., 14., 3.3)       # <--- compute jonswap spectrum
Sjon = Sbroad + Snarrow                     # <--- compute jonswap spectrum
eta  = randomphase1d(t, frqs, Sjon)         # <--- reconstruct time series
Hs = np.abs(2 + 2j)                         # <--- significant wave height
Tp = 14.                                    # <--- peak period

# --- compute wavelet transform ---
W = cwt(eta, fs, frqs)
Swav = (np.abs(W)**2).mean(1)
Swav = eta.var() * Swav  / np.trapz(Swav, x=frqs)


# --- plot spectrum and time series ---
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
#
ax1.plot(t, eta, c='k')
ax1.set_xlim((t[0]-50,t[-1]+50))
ax1.set_ylim((-Hs, Hs))
ax1.set_xlabel('$t\,\mathrm{[s]}$')
ax1.set_ylabel('$\\eta\,(t)\,\mathrm{[m]}$')
#
# --- compute density spectrum ---
for i in range(25):
    eta  = randomphase1d(t, frqs, Sjon)
    ff, SS = pwelch(eta, fs=fs, winsize=L, noverlap=0, m=1)
    print("Significant wave height: {:6.3f} m".format(4*np.trapz(SS, ff)**.5))
    if i == 0:
        ax2.plot(ff, SS, c='0.75', lw=0.5, label='Fourier')
    else:
        ax2.plot(ff, SS, c='0.75', lw=0.5)

ax2.plot(frqs, Swav, c='r', label='Wavelet')
ax2.plot(frqs, Sjon, c='k', label='Jonswap')
ax2.set_xlabel('$f\,\mathrm{[Hz]}$')
ax2.set_ylabel('$S\,(f)\,\mathrm{[m^2/Hz]}$')
ax2.set_xlim((0, .5))
ax2.set_ylim((0, 12.5))
ax2.legend(loc=1, frameon=False, ncol=1)
#
fig.subplots_adjust(hspace=.4)
fig.savefig('seasurface_jonswap.png')

# }}}
# --- end of file ---
