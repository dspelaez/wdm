#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     test_wdm_lakes.py
Created:  2017-04-03 17:53
Modified: 2017-04-03 17:53

Purpose:
    This program is used to test the WDM using the sea surface
    elevation data from the lakes in Canada, provided directly
    by Mark Donelan.

"""

# --- import libs ---
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
#
from tools.graphs import polar_spectrum
from tools.spectra import dirspec, randomphase2d
from tools.wdm import (reg_array, coarray, wavelet_spectrogram, klcomponents,
                       fdir_spectrum, kdir_spectrum, smooth, psd)
#
import os


# --- geometric array ---
angle = np.array([0., 33., 105., 177., 249., 321.])
radii = np.array([0., 0.25, 0.25, 0.25, 0.25, 0.25])
x = radii * np.cos(angle * np.pi/180.)
y = radii * np.sin(angle * np.pi/180.)
# coarray(x, y, plot=True, filename='pentagon_coarray_lakes.png')

# --- load sea surface elevation data ---
tag = 'lakes'
filename = '../data/seasurface.%s' % tag 
if os.path.exists(filename):
    A = np.genfromtxt(filename)[:2**14,:]



# --- define some parameters ---
fs    = 4.0                             # <--- sampling frequency
dt    = 1./fs                           # <--- delta time
L     = 2**14                           # <--- length of data
t     = np.arange(0, L/fs, 1./fs)       # <--- time array
dirs  = np.linspace(0, 359, 360)        # <--- array of directions


# --- compute directional spreading from wdm ---
#
# wavelet coefficients
omin, omax, nvoice = -4, 1, 4
frequencies, coefficients = wavelet_spectrogram(A, fs, omin, omax, nvoice, mode='TC98')

# compute components of wavenumber
k, l = klcomponents(frequencies, coefficients, x, y)

# compute power density from wavelets coefficients
delta_frequencies = frequencies * np.log(2**(1./nvoice))
power_wavelet = np.abs(coefficients) ** 2 * delta_frequencies[:,None,None]

# compute directional spectrum and spreading
E_donelan, D = fdir_spectrum(frequencies, dirs, power_wavelet.mean(2), k, l)
E_donelan = E_donelan / delta_frequencies

# compute wavenumber spectrum
kmax = 6.
kfac = 120. / kmax
wavenumbers = np.arange(1, kmax*kfac+1) / kfac
K_donelan = kdir_spectrum(wavenumbers, dirs, power_wavelet.mean(2), k, l)

# smooth
E_donelan = smooth(E_donelan, ws=(15,1))
K_donelan = smooth(K_donelan, ws=(15,1))

# --- compute integrated spectra  ---
S_fourier = psd(A, frequencies, fs=fs, winsize=L/4, noverlap=L/8, m=3)
S_donelan = np.trapz(E_donelan, x=dirs*np.pi/180., axis=0)

# --- plot directional spectra for each one ---
fig = plt.figure(figsize=(6.2, 3.2))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
#
ax1.loglog(frequencies, S_fourier, 'r', label='Fourier')
ax1.loglog(frequencies, S_donelan, 'g', label='Wavelet')
ax1.loglog(frequencies[frequencies > .5], 1E-2*frequencies[frequencies > .5]**-5)
ax1.text(1.1, .01, '$f^{-5}$')
ax1.legend(loc=0)
#
polar_spectrum(frequencies, dirs, E_donelan, fmax=2, ax=ax2, cbar=False, label=True, smin=-4)
#
ax1.set_xlabel('$f\,\mathrm{[Hz]}$')
ax1.set_ylabel('$S(f)\,\mathrm{[m^2 / Hz]}$')
ax2.set_ylabel('')
#
fig.savefig('directional_spectra_%s.png' % tag)


# --- plot wavenumber spectra for each one ---
fig = plt.figure(figsize=(6.2, 3.2))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
#
ax1.loglog(wavenumbers, np.trapz(K_donelan/40, x=dirs*np.pi/180., axis=0), 'g')
ax1.loglog(wavenumbers[wavenumbers > 1], 1E-2*wavenumbers[wavenumbers > 1]**-4)
# ax1.text(1.1, .01, '$f^{-5}$')
# ax1.legend(loc=0)
#
polar_spectrum(wavenumbers, dirs, K_donelan/40, fmax=4, ax=ax2, cbar=False, 
               label=True, smin=-4, is_wavenumber=True)
print(A.var() / np.trapz(np.trapz(K_donelan, x=dirs*np.pi/180., axis=0), x=wavenumbers))
#
ax1.set_xlabel('$\\kappa\,\mathrm{[Hz]}$')
ax1.set_ylabel('$S(\\kappa)\,\mathrm{[m^3 / rad]}$')
ax2.set_ylabel('')
#
fig.savefig('wavenumber_spectra_%s.png' % tag)
# --- }}}
