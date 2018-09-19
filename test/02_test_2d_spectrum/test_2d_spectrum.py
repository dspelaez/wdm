#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     2d_spectrum.py
Created:  2017-03-18 16:56

Porpuse:
    This test computes the directional wave spectrum based on a JONSWAP shape
    assings a directional form using a sech2 or cos2s and then constructs the
    surface elevation time series in two dimensions. Finally the power spectrum
    is computed from the surface elevation and compared with the orginal one.
"""

# --- import libs ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
#
from wdm.spectra import dirspec, randomphase2d, polar_spectrum
#
plt.ion()

# --- define parameters --- {{{
fs   = 5.0                                             # <--- sampling frequency
L    = 2**12                                           # <--- length of data
t    = np.arange(0, L/fs, 1./fs)                       # <--- time array of 13.6 mins
frqs = np.fft.rfftfreq(L, 1./fs)                       # <--- all fourier frequencies
dirs = np.linspace(0, 359, 360)                        # <--- array of directions
E1   = dirspec(frqs, dirs, 2.,  6.,  45., 'cos2s',  1) # <--- 2d spectrum
E2   = dirspec(frqs, dirs, 2., 14., 225., 'cos2s', 20) # <--- 2d spectrum
E    = E1 + E2                                         # <--- sum two systems
# }}}


# --- directional spreading --- {{{
S = np.trapz(E, x=dirs*np.pi/180., axis=0)
D = E / (S[None,:] + 1E-30)

# --- plot 2d-spectrum ---
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_axes([0.1,0.65, 0.8, 0.35])
ax2 = fig.add_axes([0.2,0.1,  0.55,0.4 ])

# plot integrated directional spreading
ffp = frqs * 12.
for j in [0.5, 1., 2., 3.]:
    ix = np.argmin(np.abs(ffp - j))
    ax1.plot(dirs, D[:,ix], label='$f/f_p = %.1f$' % (ffp[ix]))
#
ax1.set_xlabel('$\\theta$')
ax1.legend(loc=0)
ax1.set_xticks(np.arange(0,360+45,45))
ax1.set_xlim((0,360))

# plot 2d spectrum
polar_spectrum(frqs, dirs, E, smin=-3, fmax=.4, ax=ax2, cbar=True, label=True)
ax2.set_ylabel('')
fig.subplots_adjust(wspace=.3, hspace=.3)
fig.savefig('jonswap_bimodal.png')
# }}}


# --- compare sea surface elvation in three points --- {{{
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#
ax2.plot(frqs, S, 'k')
#
eta = randomphase2d(frqs, dirs, E)
points = ((0,0),(1,1),(500,500))
for x0, y0, in points:
    #
    # generate seasurface
    eta1d = eta(t, x=x0,  y=y0)
    #
    # compute spectrum
    ff, SS = signal.welch(eta1d, fs=fs, nperseg=L/4, noverlap=L/8, window='hanning')
    #
    # plot results
    ax1.plot(t, eta1d)
    ax2.plot(ff, SS)
#
ax2.set_xlim((0,.5))
# }}}


# --- compute sea surface elevation --- {{{
# truncate values of frequency for speedup the code
flim = 3 / 8.
E_trucated     = E[::5, frqs<flim]
frqs_truncated = frqs[frqs<flim]
dirs_truncated = dirs[::5]
# get eta function
eta = randomphase2d(frqs_truncated, dirs_truncated, E_trucated)

# # define grid
nx, ny = 2000, 2000
x = np.linspace(0, nx, nx/5)
y = np.linspace(0, ny, ny/5)
eta2d = eta(t=0, x=x[None,:], y=y[:,None])

# --- plot sea surface elevation in a map ---
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(1,1,1)
ax.pcolormesh(x, y, eta2d, vmin=-1, vmax=1, cmap=plt.cm.Blues)
ax.set_xticks([0, 500, 1000, 1500, 2000])
ax.set_yticks([0, 500, 1000, 1500, 2000])
ax.set_xlabel("$x\,\mathrm{[m]}$")
ax.set_ylabel("$y\,\mathrm{[m]}$")
fig.savefig('seasurface_simulation.png')
# }}}
