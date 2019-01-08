#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# vim:fdm=marker
#
# =============================================================
#  Copyright © 2017 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
File:     spectra.py
Created:  2017-03-15 11:23
Modified: 2017-03-18 11:23

Purpose:
    This module contains a class with functions to handle
    with spectra of wind generated waves.
"""

# --- import libs ---
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# jonswap funcion {{{
def jonswap(frqs, Hs, Tp, gamma=3.3):
    """
    This function computes the shape of the JONSWAP spectrum for
    a given peak frequency, significant wave heigth and peak
    enhancement factor.
    
                        2       /              4  \
               alpha * g       |         / fp \    |        r
    S(f)  =    ----------- exp | - beta | ---- |   | * gamma
                    4  5       |         \ f  /    |
              (2 pi)  f         \                 /
    
                              2
               /      (f - fp)     \
    r  =  exp |  -  --------------  |
              |            2   2    |
               \    2 sigma  fp    /
    

    The spectrum is normalized dividing by the zero-order moment
    and multipying by the squared significant wave height by 16
    
    Input:
        frqs  : [1d-array] array of frequencies
        Hs    : [float] significant wave height
        Tp    : [float] peak period
        gamma : [float] peak enhancement factor.
                        default is gamma=3.3. If gamma=1 we obtain
                        a Pearson-Moskowitz spectrum
    Output:
        S     : [2d-array] array with wave energy as a function of frq
    """
    
    # compute parameters 
    fp = 1./Tp
    alpha = 8.1E-3
    beta = 5./4.
    g = 9.8
    
    S = np.zeros_like(frqs)
    for i, f in enumerate(frqs):
        if f == 0.:
            S[i] = 0.
        else:
            num = alpha*(g**2.)
            den = ((2.*np.pi)**4) * f**5
            pm = (num / den) * np.exp(-beta * (fp/f)**4)
            sigma = 0.07 if f <= fp else 0.09
            r = np.exp(-(f - fp)**2 / (2.*(sigma**2)*(fp**2)))
            S[i] = pm * gamma**r
    
    
    # scale spectrum with zero-order moment 
    m0 = np.trapz(S, x=frqs)
    fac = (Hs**2/16.)/m0
    S = S*fac
    
    return S
# }}}

# directional spreading function {{{
def dirspec(frqs, dirs, Hs, Tp, mdir, func="cos2s", s=1):
    """
    This function computes the directional spreading funcion using two well-known
    methods: the first one is the classic cos2s shape and the second one and
    recommended is the sech2 proposed by DHH

    Input:
        frqs : Array of frequencies in Hz. Generally this must be organized
               logarithmically such that frq[i+1] =1.1*frq[i]
        dirs : Directions in degrees counterclockwise from east
        Hs   : Target significant wave height [m]
        Tp   : Target peak period [s]
        mdir : Mean direction [degrees from north]
        func : You may chose between `cos2s` o `sech2` [default cos2s]
        s    : If func=`cos2s` then s is the exponent [default s=1]

    Output:
        S    : array of energy density [m^2 / Hz / deg]

    References:
        Donelan, M. A., Hamilton, J., & Hui, W. (1985). Directional spectra of wind-generated waves.
        Philosophical Transactions of the Royal Society of London A: Mathematical, Physical and
        Engineering Sciences, 315(1534), 509-562.
        http://drs.nio.org/drs/bitstream/handle/2264/3113/Oceanogr_Indian_Ocean_1992_741.pdf?sequence=2
    """
     
    # crear vector de direcciones
    dir2 = (dirs - mdir) % 360
    dir3 = dirs - 180

    # crear especro escalar a partir de jonswap
    S_jon =  jonswap(frqs, Hs=Hs, Tp=Tp, gamma=3.3)

    # crear espectro direccional
    #
    # cos2s
    if func == "cos2s":
        #
        # funcion de dispersion direccional
        D = (2. / np.pi) * np.cos((dirs - mdir)*np.pi/180.)**(2*s)
        D[np.logical_and(dir2>90, dir2<270)] = 0.
        #
        # espectro direccional
        S = np.zeros((len(dirs), len(frqs)))
        for i in range(len(frqs)):
            for j in range(len(dirs)):
                S[j,i] = S_jon[i] * D[j]
    #
    # sech2
    if func == "sech2":
        #
        # funcion de dispersion direccional
        ffp = frqs*Tp
        beta = np.ones(len(frqs))*1.24
        beta[np.logical_and(ffp>=0.56, ffp<0.95)] = 2.61 * ffp[np.logical_and(ffp>=0.56, ffp<0.95)]**(1.3)
        beta[np.logical_and(ffp>=0.95, ffp<1.60)] = 2.28 * ffp[np.logical_and(ffp>=0.95, ffp<1.60)]**(-1.3)
        #
        # espectro direccional
        S = np.zeros((len(dirs), len(frqs)))
        for i in range(len(frqs)):
            for j in range(len(dirs)):
                S[j,i] = S_jon[i] * (beta[i] / 2) * np.cosh(beta[i] * (dir3[j])*np.pi/180.)**(-2)
        #
        ixd = np.argsort(((dir3 + mdir) % 360))
        S = S[ixd,:]
    
    # normalizar con la Hs
    m0 = np.abs(np.trapz(np.trapz(S, x=dirs*np.pi/180., axis=0), x=frqs))
    S = (S / m0) * (Hs**2 / 16)

    # variables de salida
    return S
# }}}

# function random phase 1d {{{
def randomphase1d(t, frqs, S):
    """
    This function computes the sea surface elevation from a
    one dimensional spectrum using a random phase.

    Input:
        t    : [1d-array] array of time
        frqs : [1d-array] array of frequencies 
        S    : [2d-array] energy spectrum

    Output:
        eta  : [1d-array] sea surface elevation
    """

    df = np.diff(frqs)[0]
    ampl = np.sqrt(2. * S * df)
    omega = 2. * np.pi * frqs
    phase = np.random.rand(len(frqs)) *2. * np.pi - np.pi
    # coeffs = ampl * np.exp(1j*phase)
    # eta = np.real(np.fft.irfft(coeffs, len(t)))
    # ---> the basic algotithm is <---
    eta = np.zeros_like(t)
    for j in range(len(frqs)):
       eta += ampl[j] * np.cos(omega[j]*t + phase[j])
    # <------------------------------>
    # NOTE: In this case, I need all
    #       fourier coefficients, in
    #       order to recover the same
    #       jonswap spectrum.
    # <------------------------------>

    return eta
# }}}

# function random phase 2d {{{
def randomphase2d(frqs, dirs, S):
    """
    This function computes the sea surface elevation from a
    two dimensional spectrum S(f,theta) using a random phase.
    Thera are two modes:
        - When mode='timeseries' an array of time must be
          specified as well as the coordinates of one point (x, y)
        - When mode='spatial' two arrays constituying a grid 
          must be specified, as well as the float time (one value)

    Input:
        t    : [1d-array] array of times
        frqs : [1d-array] array of frequencies 
        dirs : [1d-array] array of directions
        S    : [2d-array] energy spectrum

    Output:
        eta  : [nd-array] sea surface elevation

    TODO:
        - Check consistency of inputs

    """

    # --- compute 2d frequencies and directions (dirs, frqs)
    frq2d, dir2d = np.meshgrid(frqs, dirs)

    # --- parameters ---
    g = 9.8                                                 # <--- accel. due to grav.
    df = frqs[1] - frqs[0]                                  # <--- delta of frequencies
    dtheta = (dirs[1] - dirs[0]) * np.pi / 180.             # <--- delta of directions
    ampl = np.sqrt(2. * S * df * dtheta)                    # <--- amplitudes
    omega = 2. * np.pi * frq2d                              # <--- angular frequency
    kappa = omega ** 2 / g                                  # <--- dispersion relation
    kx = kappa * np.cos(dir2d * np.pi / 180.)               # <--- wavenumber in x
    ky = kappa * np.sin(dir2d * np.pi / 180.)               # <--- wavenumber in y
    phase = np.random.rand(*S.shape) *2. * np.pi - np.pi    # <--- random phase

    def eta(t=0, x=0, y=0):
        """
        Input:
            t: [1d-array or float] time
            x: [1d-array or float] x-coordinate
            y: [1d-array or float] y-coordinate
        Usage:
            This funcion returns sea surface elevation time series in 
            a grid or at specified point in meters

            If you want one time series at point (0,0) where t is an array:
             >> eta1d = eta(t, x=0, y=0)

            If we want evaluate at several points at time you can use:
             >> eta2d = eta(t[:,None,None], x[None,:,None], y[None,None,:])
        """

        # start loop for each spectral component
        n = 0.
        for i in range(len(dirs)):
            for j in range(len(frqs)):
                n += ampl[i,j] * np.cos(kx[i,j]*x + ky[i,j]*y - omega[i,j]*t + phase[i,j])
        return n

    return eta
# }}}

# plot dirctional spectrum in a polar graph {{{
def polar_spectrum(frqs, dirs, S, thetam=0, **kwargs):
    """
    This function plots the frequency-direction spectrum or
    wavenumber-direction spectrum in a polar plot.
    Aditionally it computes the bulk parameters Hs, Tp, mdir.

    Input:
        frqs     : Frequencies in Hz or wavenumbers in rad/m
        dirs     : Directions in degrees
        S        : Energy (dirs, frqs)
        thetam   : Rotation counter-clockwise
        kwargs   : Optional parameters
         - fmax          : max frequency or wavenumber
         - filename      : if it is given the plot is saved in a file
         - ax            : axis handler for a existing figure
         - smin          : mininim energy contourf
         - smax          : maximum energy contourf
         - is_wavenumber : if true add propper label
         - label         : if true add label with hs, tp, etc
         - cbar          : adds colorbar


    Output:
        Nothing. But a pretty plot is generated
    """

    # parametros integrales
    D_int = np.trapz(S, x=frqs, axis=1)
    S_int = np.trapz(S, x=dirs*np.pi/180., axis=0)
    #
    Hs   = 4. * np.sqrt(np.trapz(S_int, x=frqs))
    fp   = frqs[np.argmax(S_int)]
    Tp   = 1./fp
    pdir = np.mod(dirs[np.argmax(D_int)] + thetam, 360)
    fmax = kwargs.get('fmax', np.round(3.*fp, 1))

    # checkar si las direcciones son circulares
    if dirs[-1] != 360.:
        S0 = S.copy()
        dirs = np.append(dirs, 360)
        S = np.zeros((len(dirs), len(frqs)))
        S[:-1,:] = S0
        S[-1,:] = S0[0,:]


    # abir figura
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)


    # define energy limits
    with np.errstate(divide='ignore'):
        if 'smin' in kwargs:
            smin = float(kwargs['smin'])
        else:
            smin = np.ceil(np.max([-7, np.log10(S.min())]))
        #
        if 'smax' in kwargs:
            smax = float(kwargs['smax'])
        else:
            smax = np.floor(np.min([ 7, np.log10(S.max())])) + 1

    # draw frequency/wavenumber circles 
    fticks = np.append(np.linspace(-fmax, 0, 5), np.linspace(0, fmax, 5)[1:])
    for radii in fticks[fticks > 0]:
        circle = plt.Circle((0,0), radii, color="0.5",
                linestyle="dashed", fill=False)
        ax.add_artist(circle)
    
    # crear malla en el espacio frecuencial (Fx, Fy)
    F, D = np.meshgrid(frqs, dirs)
    Fx, Fy = F*np.cos(np.radians(D+thetam)), F*np.sin(np.radians(D+thetam))

    # create colormap
    colors = ["#FFFFFF", "#01DFA5", "#FE642E", "#08298A", "#01A9DB"]
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors, N=1024)
    norm = mcolors.LogNorm(vmin=10**smin, vmax=10**smax)
    
    # draw pcolor and contours
    cf = ax.pcolormesh(Fx, Fy, S, cmap=cmap, norm=norm)
    cr = ax.contour(Fx, Fy, S, np.array([.25, .5, .75, 1])*S.max(), colors="k")

    # add colorbar
    if 'cbar' in kwargs:
        if kwargs['cbar'] == True:
            cb = plt.colorbar(cf, ax=ax, cax=colorax(ax), 
                    ticks=10**np.arange(smin, smax+1))
        else:
            pass
    else:
        cb = plt.colorbar(cf, ax=ax, cax=colorax(ax), 
                ticks=10**np.arange(smin, smax+1))

    # force to be equal proportion
    ax.axis("equal")

    # draw labels N, S, W, E
    nlabel = kwargs.get('nlabel', True)
    if nlabel:
        ax.text(0.50, 0.95, "N", transform=ax.transAxes, ha="center", va="center")
        ax.text(0.95, 0.50, "E", transform=ax.transAxes, ha="center", va="center")
        ax.text(0.50, 0.05, "S", transform=ax.transAxes, ha="center", va="center")
        ax.text(0.05, 0.50, "W", transform=ax.transAxes, ha="center", va="center")

    ax.set_xticks(fticks)
    ax.set_xticklabels([f"{np.abs(i):.1f}" for i in fticks])
    #
    ax.set_yticks(fticks)
    ax.set_yticklabels([f"{np.abs(i):.1f}" for i in fticks])
    #
    ax.set_xlim([-fticks[-1], fticks[-1]])
    ax.set_ylim([-fticks[-1], fticks[-1]])
    #
    ax.margins(.05)
    
    # etiqueta de eje para saber si es numero de onda o frecuencia
    is_wavenumber = kwargs.get('is_wavenumber', False)
    if is_wavenumber:
        ax.set_xlabel("$\\kappa \, \mathrm{ [rad/m]}$")
        ax.set_ylabel("$\\kappa \, \mathrm{ [rad/m]}$")
    else:
        ax.set_xlabel("$f \, \mathrm{ [Hz]}$")
        ax.set_ylabel("$f \, \mathrm{ [Hz]}$")

    # agregar info
    if 'label' in kwargs:
        if kwargs['label']:
            if is_wavenumber:
                label = "$H_s = %.2f \, \mathrm{m}$\n$\\lambda_p = %.2f \,\mathrm{m}$\
                        \n$\\theta_p = %.1f^\circ$" % (Hs, 2*np.pi*Tp, pdir)
            else:
                label = "$H_s = %.2f \, \mathrm{m}$\n$T_p = %.2f \,\mathrm{s}$\
                        \n$\\theta_p = %.1f^\circ$" % (Hs, Tp, pdir)
            ax.text(0.01, 0.01, label, transform=ax.transAxes, ha="left", va="bottom")
        else:
            pass

    if 'filename' in kwargs:
        fig.savefig(kwargs['filename'])
# }}}

# create colorbar {{{
def colorax(ax, **kwargs):
    """Create a colobar without reducing parent axes size."""
    #
    width = "3%",
    height = "100%",
    loc = 3,          
    bbox_to_anchor = (1.02, 0., 1, 1),
    bbox_transform = ax.transAxes,
    borderpad = 0,
    #
    cax = inset_axes(ax,
            width="3%",
            height="100%",
            loc=3,          
            bbox_to_anchor=(1.02, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
    return cax
# }}}



if __name__ == "__main__":
    pass


# --- end of file ---
