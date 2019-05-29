#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# vim:fdm=marker
#
# =============================================================
#  Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
This module contains functions to apply the correction of the
data due to the BOMM inertial motion.
"""


import numpy as np
import scipy.signal as signal
import pandas as pd


# --- helper functions ---
# butterworth lowpass and highpass filter {{{
def butterworth_filter(data, fs=100, fc=None, order=5, kind="low"):
    """Butterworth low- and hihg-pass filter implementation."""

    if fc is not None:
        b, a = signal.butter(order, fc/(0.5*fs), btype=kind, analog=False)
        data_filtered = signal.filtfilt(b, a, data)
        if hasattr(data, "mask"):
            return np.ma.masked_array(data_filtered, mask=data.mask)
        else:
            return data_filtered
    else:
        return data
# }}}

# detrend with no nans {{{
def detrend(y, degree=1):
    """Nice detrend function to handle with NaNs."""
    
    # remove nans
    if hasattr(y, "mask"):
        if len(np.nonzero(y.mask)[0]) == len(y):
            raise Exception("Array is full of nans")
    else:
        ixnan = np.isnan(y)
        y[ixnan] = np.nanmean(y)

    # fit a polinomial
    x = np.linspace(0, len(y), len(y))
    p = np.polyfit(x, y, degree)

    # remove trend
    return y - np.polyval(p, x)
# }}}

# resample data to the same sampling frequency {{{
def resample(x, y):
    """Interpolates `y` data into `x` size and returns `y_new`
    
    This function uses pandas for an accurate resample. The `x` data is the
    fast signal and the `y` data is the slow signal. This function interpolates
    `y` data into the `x` data.
 
    """

    x_time = np.linspace(0, 100, len(x))
    y_time = np.linspace(0, 100, len(y))

    s = pd.Series(y, index=y_time)

    # check the number of nans
    n = len(y)
    n_nans = s.isnull().sum().max()
    if n_nans / n > 0.1:
        raise Exception("Number of NaNs is %d out of %d" % (n_nans, n))

    # drop missing values only if are less than 10 percent of the data
    # TODO: not more than 100 consecutive missing values
    s = s.dropna(how="any")

    # remove duplicate indices if they exist
    s = s[~s.index.duplicated(keep='first')]

    # sort data in ascending and reindex to the new time
    # i still dont know what is the difference between ffill/bfill
    s = s.sort_index().reindex(x_time, limit=1, method="bfill").ffill()

    # crate new dictionary for output
    if hasattr(x, "mask"):
        return np.ma.masked_array(s.values, mask=x.mask)
    else:
        return s.values
# }}}

# complementery filter in using a digital filter {{{
def complementary_filter(signal_a, signal_b, fs, fm, filter_order=2):
    """"Returns a merge between two angles signal_a and signal_b in radians."""
    
    # merge the signals using the merge frequency fm
    s = butterworth_filter(np.exp(1j*signal_a), fs, fm, filter_order, kind="low") + \
        butterworth_filter(np.exp(1j*signal_b), fs, fm, filter_order, kind="high")

    return np.angle(s)

# }}}

# integration in the frequency domain {{{
def fft_integration(signal, fs, fc=None, order=-1):
    """Intergration of a signal in the frequency domain using FFT.
    
    This function implements the integration in the time domain by means of the
    Fast Fourier Transform. It also performs a band pass filter, removing all
    the unwanted frequencies.

    Args:
        signal (array): Numpy 1d-array with the data.
        fs (float or int): Sampling frequency.
        fc (float or tuple): This is the cut-off frequency. If fc is floating or
            integer a lowpass filter is performed. If fc is a list a band
            pass-pass filter is made between the two given frequencies.
        order (integer): Indicates the order of the integration. Negative number
            indicates integration while positive indicates differentiontion (not
            implemented yet).

    Return (array): Signal integrated.
    """

    # check for nans if more than 10 percents
    nans = np.isnan(signal)
    if len(nans.nonzero()[0]) / len(signal) < 0.1:
        signal[nans] = np.nanmean(signal[~nans])
    else:
        return signal * np.nan
    
    # if order == 0 do nothing
    if order == 0:
        return signal

    # if order > 0 raise an error
    if order > 0:
        raise ValueError("Order must be a negative integer")

    # get frequency array
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # the first element of freqs array is zero, so we have
    # to discard it to avoid division by zero
    # the factor of integration is iw
    factor = 1j*2*np.pi*freqs[1:]
    
    # compute fft of the signal for the non-zero frequencies
    # and apply integration factor
    fft = np.zeros(len(freqs), 'complex')
    fft[1:] = np.fft.fft(signal)[1:] * factor**order
    
    # high pass filter
    if fc is None:
        return np.fft.ifft(fft).real
    elif isinstance(fc, float) or isinstance(fc, int):
        ix = abs(freqs) <= fc
        fft[ix] = 0.
    elif isinstance(fc, list) or isinstance(fc, tuple):
        flow, fhigh = fc
        ix = np.logical_and(abs(freqs) >= flow, abs(freqs) <= fhigh)
        fft[~ix] = 0.

    return np.fft.ifft(fft).real
# --- }}}

# rotation matrix in three dimensions {{{
def vector_rotation(U, E, units="rad"):
    """Apply a three dimensional rotation of the vector U given the angles T.
    
    Args:
        U (tuple): Components of the vector U = (u, v, w)
        E (tuple): Rotation angles corresponding to roll (x), pitch (y) and
            yaw (z), respectively. E = (phi, theta, psi)
        units (str): Flag to choose input angles between degrees or radians.

    Return (float): Components of the rotated vector.
    """

    # unpack tuples
    u, v, w = U
    phi, theta, psi = E

    # if input angles are in degrees, convert it to radians
    if units == "deg":
        phi, theta, psi = np.radians(phi), np.radians(theta), np.radians(psi)
    elif units == "rad":
        pass
    else:
        raise ValueError("units must be either def or rad")

    # check validity of angles
    # phi and psi must be between -pi and pi
    # theta musr be between -pi/2 and  pi/2

    # compute sins and cosines
    c_phi, c_theta, c_psi = np.cos(phi), np.cos(theta), np.cos(psi)
    s_phi, s_theta, s_psi = np.sin(phi), np.sin(theta), np.sin(psi)

    # apply the rotation of each components
    #
    # first component
    u_rot = (c_theta*c_psi)                     * u + \
            (s_phi*s_theta*c_psi - c_phi*s_psi) * v + \
            (c_phi*s_theta*c_psi + s_phi*s_psi) * w

    # second component
    v_rot = (c_theta*s_psi)                     * u + \
            (s_phi*s_theta*s_psi + c_phi*c_psi) * v + \
            (c_phi*s_theta*s_psi - s_phi*c_psi) * w

    # third component
    w_rot = (-s_theta)      * u + \
            (s_phi*c_theta) * v + \
            (c_phi*c_theta) * w

    return u_rot, v_rot, w_rot


# }}}

# angles_to_quaternions {{{
def angles_to_quaternions(T):
    """Returns the quaterions associatted with the angles T=(phi,theta,psi)."""

    # unpack tuples
    phi, theta, psi = T

    # compute sins and cosines
    c_phi, c_theta, c_psi = np.cos(phi), np.cos(theta), np.cos(psi)
    s_phi, s_theta, s_psi = np.sin(phi), np.sin(theta), np.sin(psi)

    # compute quaternions components
    q0 = 0.5*np.sqrt(1 + c_theta*s_psi + s_phi*s_theta*s_psi + \
            c_theta*c_psi + c_phi*c_theta)
    q1 = (s_phi*c_theta - c_phi*s_theta*s_psi + s_phi*c_psi) / (4*q0)
    q2 = (c_phi*s_theta*c_psi + s_phi*s_psi + s_theta) / (4*q0)
    q3 = (c_theta*s_psi - s_phi*s_theta*c_psi + c_phi*s_psi) / (4*q0)

    return (q1, q2, q3, q4)

# }}}

# split into flow, waves and turbulent part {{{
def split_into_three(data, fs=100, fc=(0.04,2)):
    """Split a signal into flow, waves and turbulent parts."""

    data_flow = bomm.butterworth_filter(data, fs, fc[0], kind="low")
    data_turb = bomm.butterworth_filter(data, fs, fc[1], kind="high")
    data_wave = bomm.butterworth_filter(data, fs, fc[0], kind="high") - data_turb

    return data_flow, data_wave, data_turb
# }}}

# get an array of a regular distribution of wavestaffs {{{
def wavestaff_coordinates(N, R=0.866, theta_0=-270):
    """This function returns position a regular array

    Function to get the coordinates (x, y) of and an array of wavestaffs formed by
    an regular N-vertices figured centered in (0,0) increasing counterclockwise
    Args:
        N (float): Number of vertices
        R (float): Separation
        theta_0 (float): Starting angle. 270 for the Ekinox reference frame.
    
    Returns:
        x (1d-array): x-coordinates of the array
        y (1d-array): y-coordinates of the array
    """

    theta = np.arange(0, 360, 360/N) - theta_0
    x, y  = [0], [0]
    x = np.append(x, R * np.cos(theta * np.pi / 180.))
    y = np.append(y, R * np.sin(theta * np.pi / 180.))

    return x, y
# }}}


# --- compute yaw, pitch and roll ---
# compute tilt from accelerations {{{
def tilt_from_accerometer(ax, ay, az):
    """Compute the inclination from the acceleromter signal as complex number."""

    # using the SBG rotation matrix and arctan
    phi = np.arctan(ay / az)
    theta = np.arctan(-ax / (ay*np.sin(phi) + az*np.cos(phi)))

    return phi, theta
# }}}

# compute pitch and roll using a complementary_filter {{{
def pitch_and_roll(ax, ay, az, wx, wy, wz, fs=100, fc=1/25, fm=1):
    """Euler angles using a complementary filter in frequency domain."""

    # compute pitch and roll from acclerometers
    phi_acc, theta_acc = tilt_from_accerometer(ax, ay, az)

    # compute pitch and roll from gyrospcope
    get_angle = lambda x: fft_integration(x, fs=fs, fc=fc, order=-1)
    phi_gyr, theta_gyr = get_angle(wx), get_angle(wy)

    # complementary filter
    phi = complementary_filter(phi_acc, phi_gyr, fs, fm=fm)
    theta = complementary_filter(theta_acc, theta_gyr, fs, fm=fm)

    # return data
    return phi, theta
# }}}

# orientation from magnetic north {{{
def yaw_from_magnetometer(wz, heading, fs=100, fc=1/25, fm=1):
    """Returns the yaw angle measured clockwise from north.
    
    This function computes the yaw angle which is equivalent to the buoy
    orientation. The function requires the data from the accelerometer and the
    magnetometer heading. Data are interpolated to the maximum sampling
    frquency. Uses a complementary digital filter to merge the high frequency
    gyroscope data with the low frequency magnetometer.

    Args:
        wz (float): Angular rate of change in rad/s.
        heading (float): Heading angle from magnetometer or signature in rad.
            Fortunately the heading usually follows the nautical convention,
            this means that the angle is measured clockwise which is consistent
            withe the Ekinox convention for the angles (Tait-Bryan or North,
            Easth, Down).

    Returns (float): Orientation respect to magnetic north.
    """

    # interpolate to the `wz` sampling frquency
    heading_fast = resample(wz, np.cos(heading)) + \
        1j* resample(wz, np.sin(heading))

    # compute psi angle from the magnetometre
    psi_mag = np.mod(np.ma.angle(heading_fast), 2*np.pi)
    mean_psi = np.nanmean(psi_mag)

    # compute pitch and roll from gyrospcope
    psi_gyr = fft_integration(wz, fs=fs, fc=fc, order=-1)

    # complementary filter
    psi = complementary_filter(psi_mag-mean_psi, psi_gyr, fs, fm)

    return np.mod(psi + mean_psi, 2*np.pi)
# }}}


# --- compute real position ---
# position correction {{{
def position_correction(X, A, E, fs=20, fc=1/25, q=5, full=False):
    """Correcion of the position and the surface elevation.

    This function applies the correction of the surface elevation measured by
    the wavestaffs due to the buoy inertial motion. The correction is perform
    not only to the surface elevation (z coordinate) but also for the position
    in the x-y-plane. So, the input/output is the uncorrected/corrected
    time-varying position vector of the water surface.

    The equation to perfom such correction is given by
    
                              //                 /
      x_E =   R x_B    +    R || a_B dt   +    R | curl { Om_B, x_B } dt  
                              //                 /
              ---v---    -------v-------      ------------v--------------
               x_obs          x_acc                     x_rot

    where R is a rotation matrix.

           |  cosp*cosy  sinr*sinp*cosy-cosr*siny  cosr*sinp*cosy+sinr*siny |
       R = |  cosp*siny  sinr*sinp*siny+cosr*cosy  cosr*sinp*siny-sinr*cosy |
           |   -sinp            sinr*cosp               cosr*cosp           |

    in which, `p` is pitch (theta), `r` is roll (phi) and `y` is yaw (psi)

    In the same way, the matrix of angular rate of changes is given by:

              | - dpdt siny + drdt cosp cosy |
      Omega = |   dpdt cosy + drdt cosp siny |  
              |      dydt   - drdt sinp      |          

    Args:
        X (tuple): Contains the elements of X=(x,y,z). Each component is a time
            series given in a numpy 1d array.
        A (tuple): Contains the time series of the accelerations A=(ax, ay, az).
        E (dict): Contains the time series of the euler angles
             given by E=(roll, pitch, yaw).
        fs (float): Sampling frequency of the time series.
        fc (float): Cut-ff frequency for the integration. Could be a tuple.
        q (int): Factor to decimate the accelerometer data into wavestaff data.
            This value must be 5 since the ekinox frequency is 5 times the
            wavestaff frequency. When wavestaff measured at 10 Hz q must be 10.
        full (bool): Return full values or only the corrected ones. Default False.

    Returns: Tuple with the X tuple corrected. 

    Note:
        Arrays must be clean before attempting to apply the correction.

    References:
        * Anctil Donelan Drennan Graber 1994, JAOT 11, 1144-1150
        * Drennan Donelan Madsen Katsaros Terray Flagg 1994, JAOT 11, 1109-1116
    """
    
    # substract gravity acceleration effect
    G = vector_rotation((0,0,-9.8), E)

    # apply double integration in the frequency domain
    P = (fft_integration(A[0], fs*q, fc=fc*2.5, order=-2), # TODO: WARNING
         fft_integration(A[1], fs*q, fc=fc*2.5, order=-2), # TODO: WARNING
         fft_integration(A[2], fs*q, fc=fc, order=-2))
    
    # compute the derivative of the euler angles
    D = tuple(np.gradient(e, 1/(fs*q)) for e in E)

    # decimate the high frequency signals to the given frecuency
    # note than in this function i use the following equivalences
    #   roll  --> r --> phi   --> E[0]
    #   pitch --> p --> theta --> E[1]
    #   yaw   --> y --> psi   --> E[2]
    decimate = lambda x, q: x[::q]
    E_down = tuple(decimate(e, q) for e in E) # <- Euler
    P_down = tuple(decimate(p, q) for p in P) # <- Position
    D_down = tuple(decimate(d, q) for d in D) # <- Derivative

    # compute sines and cosines
    roll, pitch, yaw = E_down
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # convert observations into the inertial frame
    x_obs, y_obs, z_obs = vector_rotation(X, (roll, pitch, yaw))
    
    # correction due to translations
    x_acc, y_acc, z_acc = vector_rotation(P_down, (roll, pitch, yaw))

    # correction due to rotation
    xB, yB, zB = X
    droll, dpitch, dyaw = D_down
    curl = (fft_integration( dpitch*xB -   dyaw*yB, fs, fc, order=-1),
            fft_integration(-dpitch*zB +   dyaw*xB, fs, fc, order=-1),
            fft_integration(  droll*yB - dpitch*xB, fs, fc, order=-1))
    x_rot, y_rot, z_rot = vector_rotation(curl, (roll, pitch, yaw))

    # compute earth-based water position
    xE = x_obs + x_acc + x_rot
    yE = y_obs + y_acc + y_rot
    zE = z_obs + z_acc + z_rot

    # return data
    if full:
        return (x_obs,y_obs,z_obs), (x_acc,y_acc,z_acc), (x_rot,y_rot,z_rot)
    else:
        return xE, yE, zE

# --- }}}


# --- run as a script ---
if __name__ == "__main__":
    pass


# --- end of file ---
