#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     wdm.py
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

# --- function to get array of a regular distribution ---
# {{{
def reg_array(N, R, theta_0):
    """
    Function to get the coordinates (x, y) of and an array of wavestaffs formed by
    an regular N-vertices figured centered in (0,0) increasing counterclockwise

    Input:
        N       : [float] Number of vertices
        R       : [float] Separation
        theta_0 : [float] Starting angle
    
    Output:
        x: [1d-array] x-coordinates of the array
        y: [1d-array] y-coordinates of the array
    """

    theta = np.arange(1, 360, 360/N) - theta_0
    x, y  = [0], [0]
    x = np.append(x, R * np.cos(theta * np.pi / 180.))
    y = np.append(y, R * np.sin(theta * np.pi / 180.))
    return x, y
# }}}

# --- function to plot the coarray ---
# {{{
def coarray(x, y, *args, **kwargs):
    """
    This function computes the co-array of an array of wavestaffs as in Young (1994)

    Input:
        x,y: [1d-array] coordinates of the geometric array
    """

    # --- check some kwargs ---
    if 'plot' in kwargs:
        flag_plot = kwargs['plot']
    else:
        flag_plot = False
    #
    if 'filename' in kwargs:
        filename = kwargs['filename']
    else:
        filename = None

    # --- compute some useful parameters ---
    if len(x) == len(y):
        # number of elements
        N = len(x)
        #
        # max distance between elements
        R = np.abs(x + 1j*y).max()
    else:
        raise ValueError('x and y must have the same length')

    # --- calcular co-array ---
    alphas = np.arange(0, 359, 45)
    for alpha in alphas:
        ii = 0
        k = [np.cos(alpha * np.pi / 180.), np.sin(alpha * np.pi / 180.)]
        d  = np.zeros((N,N,2))
        xi = np.zeros((N**2))
        for i in range(N):
            for j in range(N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d[i,j,:] = [dx, dy]
                xi[ii] = np.dot(k, [dx, dy])
                ii += 1

    # --- plot coarray ---
    if flag_plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=False)
        #
        ax1.plot(x, y, 'o', mfc='y', mec='y')
        ax1.set_xlim((-2.5*R, 2.5*R))
        ax1.set_ylim((-2.5*R, 2.5*R))
        ax1.set_xlabel('R')
        ax1.set_ylabel('R')
        ax1.set_title('Arreglo')
        ax1.set_aspect('equal')

        alphas = np.arange(0, 359, 45)
        for alpha in alphas:
            ii = 0
            k = [np.cos(alpha * np.pi / 180.), np.sin(alpha * np.pi / 180.)]
            d  = np.zeros((N,N,2))
            xi = np.zeros((N**2))
            for i in range(N):
                for j in range(N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d[i,j,:] = [dx, dy]
                    xi[ii] = np.dot(k, [dx, dy])
                    ii += 1
            
            ax2.plot(k[0]*xi, k[1]*xi, '.k')

        ax2.set_xlim((-2.5*R, 2.5*R))
        ax2.set_ylim((-2.5*R, 2.5*R))
        ax2.set_xlabel('R')
        ax2.set_title('Co-arreglo')
        ax2.set_aspect('equal')
        #
        if filename is not None:
            fig.savefig(filename)
        else:
            return fig, ax1, ax2

    else:
        return xi
# }}}



# --- define mother wavelet ---
# {{{
def morlet(scale, omega, omega0=6.):
    return (np.pi ** -.25) * np.exp(-0.5 * (scale * omega - omega0) ** 2.)
# }}}

# --- continuous wavelet transform ---
# {{{
def cwt(x, fs, freqs, mother=morlet):
    """
    This function compute the continuous wavelet transform
    
    Input:
        x      : time series
        fs     : sampling frquency [Hz]
        freqs  : array of frequencies
        mother : function to compute the cwt

    Output:
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
    return fac *  np.fft.ifft(fft[None,:] * w, ntime)
# }}}

# --- continuous wavelet transform - bertran chapron ---
# {{{
def cwt_bc(x, fs, freqs, mother=morlet):
    """
    This function compute the continuous wavelet transform
    
    Input:
        x      : time series
        fs     : sampling frquency [Hz]
        freqs  : array of frequencies
        mother : function to compute the cwt

    Output:
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


# --- function to compute wavelet spectrograms of each wavestaff ----
# {{{
def wavelet_spectrogram(A, fs, omin=-6, omax=1, nvoice=16, mode='TC98'):
    """This function computes the wavelet spectrogram of an array of timeseries."""

    # --- define scales ---
    freqs = 2.**np.linspace(omin, omax, nvoice * abs(omin-omax)+1)

    # --- compute useful variables ---
    ntime, npoints = A.shape 
    nfrqs = len(freqs)

    # --- compute wavelet coefficients ---
    W = np.zeros((nfrqs, ntime, npoints), 'complex')
    for i in range(npoints):
        if mode == "TC98":
            W[:,:,i] = cwt(A[:,i], fs, freqs)
        elif mode == "BC":
            W[:,:,i] = cwt_bc(A[:,i], fs, freqs) * np.sqrt(1.03565 / nvoice) 
        else:
            raise ValueError("Mode must be TC98 or BC")

    # --- return wavelets coefficients
    return freqs, W
# }}}

# --- function to compute wavenumber from spectrogram and array geometry ---
# {{{
def klcomponents(W, x, y, *args, **kwargs):
    """
    This function computes the directional wave spectrum using the
    Wavelete Directional Method proposed by Donelan et al. (1996)

    Input:
        W    : [2d-array] Wavelet coefficients
        x    : [2d-array] time varying x-coordinate of wavestaffs
        y    : [2d-array] time varying y-coordinate of wavestaffs

    Output:
        k,l  : [2d-array] least square estimation of kk components

    """

    # --- compute phase and power for each frequency and time
    power = np.abs(W)**2
    power = power.mean(2)
    phase = np.arctan2(-W.imag, W.real) # -> cartesian towards

    # --- dimensions ---
    nfrqs, ntime, npoints = W.shape
    neqs    = int(npoints * (npoints-1) / 2)
    npairs  = int(neqs * (neqs-1) / 2)

    # --- loop for N unique pairs of equations ---
    X = np.zeros((ntime, neqs, 2))          # <--- matrix of distances
    Dphi = np.zeros((nfrqs, ntime, neqs))   # <--- vector of phase diffs
    #
    ij = 0
    for i in range(npoints-1):
        for j in range(i+1, npoints):
            #
            # distances between pairs for each time
            for k in range(ntime):
                dx = x[k,j] - x[k,i]
                dy = y[k,j] - y[k,i]
                X[k,ij,:] = [dx, dy] 
            #
            # difference of phases
            Dphi[:,:,ij] = phase[:,:,j] - phase[:,:,i]
            #
            # acumulate counter
            ij += 1

    # --- contrains of phase differences ---
    # Dphi[Dphi >  np.pi] -= 2. * np.pi
    # Dphi[Dphi < -np.pi] += 2. * np.pi
    limit = kwargs.get('limit', np.pi)
    mask  = kwargs.get('mask',  np.nan)
    Dphi[np.abs(Dphi) > limit] = mask

    # --- least square estimation of vector kk=(k, l) ---
    #     the LSR of kk is given by:
    #       kk^LS = (X^T X)^-1 X^T phi
    #
    # function to compute wavenumber vector
    def kk(X, Dphi):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Dphi)

    # TODO: define function to solve point to point
    #
    # evaluate function at each f-t point.
    k = np.zeros((nfrqs, ntime))
    l = np.zeros((nfrqs, ntime))
    for i in range(nfrqs):
        for j in range(ntime):
            k[i,j], l[i,j] = kk(X[j], Dphi[i,j,:])
    
    # return k and l
    return k, l
# }}}

# --- function to compute directional spreading function ---
# {{{
def directional_spreading(frqs, power, k, l):
    """
    Input:
        frqs       [1d-array] : Frequencies in Hz.
        power(f,t) [2d-array] : Power wavelet spectrum.
        k(f,t)     [2d-array] : X-component of wavenumber array.
        l(f,t)     [2d-array] : Y-component of wavenumber array.
    
    Output:
        D(f,dirs) [2d-array] : Directional spreading function in radians
    """

    # array of directions each degree
    dirs = np.arange(0, 360, 1)

    # compute magnitude and direction of wavenumber
    kappa = np.abs(k + 1j*l)
    theta = np.arctan2(l, k) # -> angle points the direction towards waves goes

    # round angles to a resolution of 1 degree and correct
    # angle to be measured counterclockwise from east
    theta[theta <= 0] += 2. * np.pi
    theta_degrees = np.round(theta * 180./np.pi)

    # length of arrays
    nfrqs, ntime = power.shape
    ndirs = len(dirs)

    # loop for each frequency
    D = np.zeros((ndirs,nfrqs), 'float')
    for j in range(nfrqs):
        
        # loop for each direction
        for i in range(360):
            ix = theta_degrees[j,:] == i
            dd = len(ix.nonzero()[0])
            weight = np.mean(power[j,ix]) if dd != 0 else 0.
            D[i,j] = dd * weight

    # normalize to satisfy int(D) = 1 for each direction
    m0 = np.trapz(D, x=dirs*np.pi/180, axis=0)
    D = D / m0[None,:]
    return D
# }}}

# --- function to smooth 2d arrays ---
# {{{
def smooth(F, ws=(5,1)):
    """
    This function takes as an argument the directional spectrum or
    another 2d function and apply a simple moving average with
    dimensions given by winsize.

    For example, if a we have a directional spectrum E(360,64) and 
    ws=(5,2) the filter acts averging 10 directiona and 2 frequencies.

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

# --- function to interpolate spectrogram at specific frequencies ---
# {{{
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

# --- function to compute frequency - direction spectrum
# --- {{{
def fdir_spectrum(A, x, y, fs=10, omin=-6, omax=2, nvoice=16, ws=(30,1)):
    """Simple and ugly implementation of Wavelet Directional Method.

    Args:
        A (array): Surface elevation in the array.
        x (array): Time-varying x position of each probe.
        y (array): Time-varying y position of each probe.
        fs (float): Sampling frequency.
        omin (float): Min octave.
        omax (float): Max octave.
        nvoice (float): Number of voices.
        ws (tuple): Number of directions and frequencies to smooth.

    Returns:
        Frequceny-direction wave spectrum.

    """

    ntime, nprobes = A.shape
    nfft = int(2**np.floor(np.log2(ntime)))
    nperseg = int(nfft / 4)

    # obtain wavelet spectrogram for each gauge
    frqs, coefs = wavelet_spectrogram(A, fs, omin, omax, nvoice, mode='TC98')

    # compute components of wavenumber
    k, l = klcomponents(coefs, x, y, limit=np.pi)

    # compute power density from wavelets coefficients
    dirs = np.arange(0, 360, 1)
    power = np.mean(np.abs(coefs) ** 2, axis=2)

    # compute fourier spectrum and interpolate to wavelet frequencies
    Pxx = np.zeros((int(nperseg/2+1), nprobes))
    for j in range(nprobes):
        f, Pxx[:,j] = signal.welch(A[:,j], fs, "hann", nperseg)
    S = interpfrqs(Pxx.mean(1)[1:], f[1:], frqs)

    # compute directional spreading function and frequency direction spectrum
    D = directional_spreading(frqs, power, k, l)
    E = S[None,:] * D

    # smooth
    D_smoothed = smooth(D, ws)
    E_smoothed = smooth(E, ws)

    return frqs, dirs, E_smoothed, D_smoothed
# --- }}}



if __name__ == "__main__":
    pass


# --- end of file ----
# vim:foldmethod=marker
