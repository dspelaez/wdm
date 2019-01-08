#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     main.py
Created:  2017-03-19 11:13
Modified: 2017-04-10 18:02

Purpose:
    This module contain classes and functions to perform the Wavelet Directional
    Method to an array of WaveStaffs in order to get the wavenumber-direction
    wave spectrum. This method was proposed by Donelan et al. (1996)

References:
    Donelan, M. A., Drennan, W. M., & Magnusson, A. K. (1996). Nonstationary
    analysis of the directional properties of propagating waves. Journal of
    Physical Oceanography, 26(9), 1901-1914.

    Donelan, M., Babanin, A., Sanina, E., & Chalikov, D. (2015). A comparison of
    methods for estimating directional spectra of surface waves. Journal of
    Geophysical Research: Oceans, 120(7), 5040-5053.
    
    Hauser, D., Kahma, K. K., Harald E. Krogstad, Susanne Lehner, Jaak Monbaliu
    and Lucy R. Wyatt (2003). Measuring and analysis the directional spectrum of
    ocean waves. URL:
    http://projects.knmi.nl/geoss/cost/Cost_Action_714_deel_1.pdf
        
    Hampson, R. W. (2008). Video-based nearshore depth inversion using WDM
    method. URL: http://www1.udel.edu/kirby/papers/hampson-kirby-cacr-08-02.pdf
    """

# --- import libs ---
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import os

from .wavelet import getfreqs, morlet, cwt, cwt_bc
from .core import (position_and_phase, compute_wavenumber,
                   directional_spreading) 


# wavelet spectrograms of each wavestaff {{{
def wavelet_spectrogram(A, fs, omin=-6, omax=2, nvoice=32, mode='TC98'):
    """This function computes the wavelet spectrogram of an array of timeseries.
    
    Args:
        A (NxM Array): Surface elevation for each N wavestaff and M time.
        fs (float): Sampling frequency.
        omin (int): Minimun octave. It means fmin = 2^omin
        omax (int): Maximum octave. It means fmax = 2^omax
        nvoice (int): Number of voices. It means number of points between each
            order of magnitud. For example between 2^-4 and 2^-3 will be nvoice
            intermediate points.
        mode (str): String to define if CWT is computing follong Torrence and
            Compo (1998) method (TC98) or Bertrand Chapron's (BC).
    
    """

    # define scales 
    freqs = getfreqs(omin, omax, nvoice)

    # compute length of variables 
    ntime, npoints = A.shape 
    nfrqs = len(freqs)

    #  compute wavelet coefficients
    W = np.zeros((nfrqs, ntime, npoints), dtype='complex')
    for i in range(npoints):
        if mode == "TC98":
            W[:,:,i] = cwt(A[:,i], fs, freqs)
        elif mode == "BC":
            W[:,:,i] = cwt_bc(A[:,i], fs, freqs) * np.sqrt(1.03565 / nvoice) 
        else:
            raise ValueError("Mode must be TC98 or BC")

    return freqs, W
# }}}

# fill possible nan values {{{
def fill_nan(E):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B# }}}

# smooth 2d arrays {{{
def smooth(F, ws=(5,1)):
    """
    This function takes as an argument the directional spectrum or
    another 2d function and apply a simple moving average with
    dimensions given by winsize.

    For example, if a we have a directional spectrum E(360,64) and 
    ws=(10,2) the filter acts averging 10 directiona and 2 frequencies.

    Input:
        F  : Input function
        ws : Windows size

    Output:
        F_smoothed : Function smoothed

    """

    # define window
    nd, nf = ws
    frqwin = np.ones(nf)
    dirwin = signal.hamming(nd)
    window = frqwin[None,:] * dirwin[:,None]
    window = window / window.sum()
    
    # permorm convolution and return output
    return signal.convolve2d(F, window, mode='same', boundary='wrap')
# }}}

# interpolate spectrogram at specific frequencies {{{
def interpfrqs(S, frqs, new_frqs):
    """
    This function remap the log-spaced frequencies into linear frequencies
    
    Input:
        W       : Wavelet coefficiets. Dimensions W(nfrqs, ntimes, npoints)
        frqs     : log-spaced frequencies
        new_frqs : linear-spaced frequencies
    
    """
    return interpolate.interp1d(frqs, S, fill_value='extrapolate')(new_frqs)
# }}}

# frequency - direction spectrum {{{
def fdir_spectrum(A, x, y, fs, limit=np.pi, omin=-6, omax=2,
                  nvoice=32, ws=(30,1)):

    """Computes the frequency-direction spectrum using the WDM method.

    Args:
        A (array): Surface elevation for each probe.
        x (array): Time-varying x position of each probe.
        y (array): Time-varying y position of each probe.
        fs (float): Sampling frequency.
        limit (float): Limit for phase differences
        omin (float): Min octave.
        omax (float): Max octave.
        nvoice (float): Number of voices.
        ws (tuple): Number of directions and frequencies to smooth.

    Returns:
        Frequceny-direction wave spectrum.

    """

    # check dimensions
    ntime, npoints = A.shape
    nfft = int(2**np.floor(np.log2(ntime)))
    nperseg = int(nfft / 4)

    # check if x and y varying in time, if not, repeat its value ntimes
    if x.ndim == 2 and y.ndim == 2:
        pass
    else:
        x, y = np.tile(x,(ntime,1)), np.tile(y, (ntime,1))

    # obtain wavelet frequencies and coefficients for each gauge
    frqs, wcoefs = wavelet_spectrogram(A, fs, omin, omax, nvoice, mode='TC98')

    # compute phase diffrenence and position
    # here we use the fortran subroutine `position_and_phase` in core.f90
    neqs = int(npoints * (npoints-1) / 2)
    XX, Dphi = position_and_phase(wcoefs, x, y, neqs)
    if limit:
        min_phase = 0.0
        Dphi[Dphi == 0] = min_phase
        Dphi[Dphi >  limit] = min_phase
        Dphi[Dphi < -limit] = min_phase

    # compute components of wavenumber
    kx, ky = compute_wavenumber(XX, Dphi)
    
    # compute power density from wavelet coefficients
    dirs = np.arange(0, 360)
    power = np.mean(np.abs(wcoefs) ** 2, axis=2)

    # compute fourier spectrum and interpolate to wavelet frequencies
    Pxx = np.zeros((int(nperseg/2+1), npoints))
    for j in range(npoints):
        f, Pxx[:,j] = signal.welch(A[:,j], fs, "hann", nperseg)
    S = interpfrqs(Pxx.mean(1)[1:], f[1:], frqs)

    # compute directional spreading function
    D = directional_spreading(wcoefs, kx, ky)

    # check nans if they exist fill
    if np.isnan(D).any():
        print(frqs[np.isnan(D.mean(0))])
    
    # frequency direction spectrum
    E = S[None,:] * D

    # smooth
    if ws:
        D_smooth = smooth(D, ws)
        E_smooth = smooth(E, ws)
        return frqs, dirs, E_smooth, D_smooth
    else:
        return frqs, dirs, E, D
# --- }}}


if __name__ == "__main__":
    pass


# --- end of file ----
# vim:foldmethod=marker
