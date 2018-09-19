#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""This module contains functions to compute and plot the co-array"""

import numpy as np
import matplotlib.pyplot as plt


# --- function to get array of a regular distribution --- {{{
def reg_array(N, R, theta_0):
    """Thisn function returns position a regular array

    Function to get the coordinates (x, y) of and an array of wavestaffs formed by
    an regular N-vertices figured centered in (0,0) increasing counterclockwise

    Args:
        N (float): Number of vertices
        R (float): Separation
        theta_0 (float): Starting angle
    
    Returns:
        x (1d-array): x-coordinates of the array
        y (1d-array): y-coordinates of the array
    """

    theta = np.arange(1, 360, 360/N) - theta_0
    x, y  = [0], [0]
    x = np.append(x, R * np.cos(theta * np.pi / 180.))
    y = np.append(y, R * np.sin(theta * np.pi / 180.))

    return x, y
# }}}

# --- function to plot the coarray --- {{{
def co_array(x, y, *args, **kwargs):
    """Coarray for a given geometrical array.

    This function computes the co-array of an array of wavestaffs as in Young (1994)

    Args:
        x (1d-array): x coordinates of the geometric array
        y (1d-array): y coordinates of the geometric array
        plot (bool): flag to plot or not
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

