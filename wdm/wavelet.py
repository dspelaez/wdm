#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.


"""This module contains functions to comput the CWT for a given signal."""


# import libs
import numpy as np

# compute wavelet frequencies {{{
def getfreqs(omin, omax, nvoice):
    """Returns the frequencies arrays for the wavelet spectrum.

    Args:
        omin (int): Minimun octave. It means fmin = 2^omin
        omax (int): Maximum octave. It means fmax = 2^omax
        nvoice (int): Number of voices. It means number of points between each
            order of magnitud. For example between 2^-4 and 2^-3 will be nvoice
            intermediate points.

    Returns: Array of frequencies logarithmic distributed.

    """
    return 2.**np.linspace(omin, omax, nvoice * abs(omin-omax)+1)
# }}}

# define mother wavelet {{{
def morlet(scale, omega, omega0=6.):
    """Returns the Morlet mother wavelet for a given scale array."""

    return (np.pi ** -.25) * np.exp(-0.5 * (scale * omega - omega0) ** 2.)
# }}}

#  continuous wavelet transform as torrence and compo {{{
def cwt(x, fs, freqs, mother=morlet):
    """
    This function compute the continuous wavelet transform
    
    Args:
        x      : time series
        fs     : sampling frquency [Hz]
        freqs  : array of frequencies
        mother : function to compute the cwt

    Returns:
        freqs, W:
    """

    # compute scales
    if mother == morlet:
        f0 = 6.
        flambda = (4 * np.pi) / (f0 + np.sqrt(2. + f0 ** 2.))
    else:
        raise NotImplementedError("Only Mortet was defined so far.")

    # scale
    scale = 1. / (freqs * flambda)

    # number of times and number of scales
    ntime  = len(x)
    nscale = len(scale)

    # fourier frequencies
    omega  = 2 * np.pi * np.fft.fftfreq(ntime, 1./fs)

    # loop for fill the window and scales of wavelet
    k = 0
    w = np.zeros((nscale, ntime))
    for k in range(nscale):
        w[k,:] = np.sqrt(scale[k] * omega[1] * ntime) * mother(scale[k], omega)

    # fourier transform of signal
    fft = np.fft.fft(x)

    # convolve window and transformed series
    fac = np.sqrt(2 / fs / flambda)
    return fac * np.fft.ifft(fft[None,:] * w, ntime)
# }}}

# continuous wavelet transform as bertran chapron {{{
def cwt_bc(x, fs, freqs, mother=morlet):
    """This function compute the continuous wavelet transform
    
    Args:
        x      : time series
        fs     : sampling frquency [Hz]
        freqs  : array of frequencies
        mother : function to compute the cwt

    Returns:
        freqs, W:
    """
    
    # number of times and number of scales
    ntime  = len(x)
    nscale = len(freqs)

    # mother function
    def morlet(s):
        nu  = s * fs * np.arange(1, ntime/2+1) / ntime
        return np.exp(-1./np.sqrt(2) * ((nu - 1)/0.220636)**2.)

    # loop for fill the window and scales of wavelet
    k = 0
    w = np.zeros((nscale, int(ntime/2)))
    for k in range(nscale):
        w[k,:] = morlet(1./freqs[k])

    # real fourier transform of signal
    fft = np.fft.fft(x)
    fft = fft[1:int(ntime/2)+1]
    fft[0] /= 2.

    # convolve window and transformed series
    return np.fft.ifft(2. * fft[None,:] * w, ntime)
# }}}
