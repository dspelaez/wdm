#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""

"""

import numpy as np
import scipy.signal as signal
import datetime as dt
import yaml
import sys
import os
#
import wdm as wdm
import motion_correction as motcor
from read_data import ReadRawData


class ProcessingData(object):

    """
    This class contains funcion to get the directional wave spectra from the
    BOMM SSD using the Wavelet Directional Method (WDM)

    Usage:

        # define the path of the metadata file
        >> metafile = "../metadata/bomm1_its.yml"
        
        # define the specific date
        >> date = dt.datetime(2017,11,17,0,0)

        # create instance of the main class
        >> p = ProcessingData(metafile, date)

        # load the data, perform motion_correction and get_spectrum
        >> p.read_data()
        >> p.get_directional_spectrum()
        >> p.write_directional_spectrum()

        # [optional] get the wave parameters
        >> p.get_wave_parameters()
    """

    # private methods {{{
    def __init__(self, metafile, date):
        """Function to initialize the class.

        Args:
            metafile (str): filename containing the metadata
            date (datetime): datetime object
        """

        self.metafile = metafile
        self.date = date

        # load metadata
        with open(metafile, "r") as f:
            self.metadata = yaml.load(f)


    # check for nans in input variables
    def _check_nans(self, dic, limit=0.3):
        """Check if some of our dictionaries has more nans than wanted."""

        valid = True
        for k,v in dic.items():
            try:
                number_of_nans = len(np.nonzero(v.mask)[0])
                if (number_of_nans / len(v)) >= limit:
                    valid = False
                    return valid
            except AttributeError as e:
                pass
        return valid

    # }}}

    # read data {{{
    def read_data(self):
        """Function to read the raw data from the BOMM SDD."""

        # create instance of ReadRawData
        r = ReadRawData(self.metafile)

        # load data as dictionaries
        self.ekx = r.read("ekinox", self.date)
        self.wav = r.read("wstaff", self.date)
        self.sig = r.read("signature", self.date)
        self.met = r.read("maximet", self.date)
        self.gps = r.read("gps", self.date)

        self.results = {}
    # }}}


    # wavenumber {{{
    def wavenumber(self, f, d=100, mode="hunt"):
        """
        mode = "exact"
        --------------
        Calculo del numero de onda usando la relacion de dispersion de la teoria
        lineal resuelta con un metodo iterativo (punto fijo) sin tener en cuenta
        el efecto doppler de las corrientes en la frecuencia de las olas.
                2
              w    =   g * k * tanh(k*h)    --->    w  = 2*pi*f

        mode = "hunt"  (default)
        ------------------------
        Calculo del numero de onda usando la aproximacion empirica propuesta por Hunt 1979
                  2        2             y
              (kh)    =   y   +  --------------------
                                       6
                                     ----      n
                                 1 + \    d   y
                                     /     n
                                     ----
                                     n = 1
                   2
              y = w  h  / g

              d0 = [0.666, 0.355, 0.161, 0.0632, 0.0218, 0.0065]
            """

        if d < 0:
            raise ValueError("Depth must be positive")

        if mode == "exact":
            #
            tol = 1e-9
            maxiter = 1000000
            g = 9.8
            w = 2.* np.pi * f
            k0 = (w**2.)/g
            for cnt in range(maxiter):
                k = (w**2)/(g*np.tanh(k0*d))
                k0 = k
                if all(abs(k - k0) >= tol):
                    return k0
            return k

        elif mode == "hunt":
            #
            d0 = [0.666, 0.355, 0.161, 0.0632, 0.0218, 0.0065]
            g = 9.8
            w = 2.* np.pi * f
            y = (w**2)*d/g
            #
            poly = np.zeros_like(f)
            for n, dn in enumerate(d0):
                poly = poly + dn * y**(n+1)
            #
            k = np.sqrt(y**2 + y/(1 + poly))/d

            return k
            #
        else:
            raise ValueError("`mode` must br `hunt` o `exact`")
    # }}}

    # fourier spectrum {{{
    def welch(self, x, fs, nfft=512, overlap=128):
        """Computes the Fourier periodograms ignoring segments with NaNs."""

        # check how if all data is nan
        n = len(x)
        nans = len(np.where(np.isnan(x))[0])
        if n == nans:
            raise Exception("Array is full of NaNs.")

        # loop for each segment
        S = []
        for j in np.arange(0, n-nfft+overlap, overlap):
            arr = x[j:j+nfft]
            nans = len(np.where(np.isnan(arr))[0])
            if nans == 0 and len(arr) == nfft:
                f, S0 = signal.welch(arr, fs, window="hann", nperseg=nfft)
                S += [S0]

        return f, np.mean(S, axis=0)
    # }}}

    # compute wave parameters {{{
    def wave_parameters(self, frqs, dirs, E):
        """"Return basic bulk wave parameters from the directional wave spectrum."""

        # TODO: check for nans

        # integrate directional wave specrtum
        dirs = np.radians(dirs)
        S_int = np.trapz(E, x=dirs, axis=0)
        D_int = np.trapz(E, x=frqs, axis=1)

        # computes m,n oder moments of the spectrum
        # indices <<mn>> represents the exponents of f and S respectivaly
        m = lambda n, p: np.trapz((frqs**n)*(S_int**p), x=frqs)
        
        # compute basic parameters
        Hm0 = 4.0 * np.sqrt(m(0,1))
        Tp1 = m(0,4) / m(1,4)
        #
        # compute directional params
        m = lambda n, p: np.trapz((dirs**n)*(D_int**p), x=dirs)
        # pDir = np.mod(dirs[np.argmax(D_int)], 2*np.pi)
        pDir = m(1,4) / m(0,4)
        mDir = m(1,1) / m(0,1)

        return Hm0, Tp1, np.degrees(pDir), np.degrees(mDir)

    # }}}

    # compute stokes drift profile {{{
    def stokes_drift(self, f, S, z=-np.logspace(-5,2,50)):
        """Compute stokes drift profile as Breivik et al 2016 eq5."""
        
        # angular frequency and spectrum in right units
        g = 9.8
        k = self.wavenumber(f, 100) 
        w = 2*np.pi * f
        Sw = S / (2*np.pi)
        
        fac = 2 / g
        if isinstance(z, float) or isinstance(z, int):
            dummy = w**3 * Sw * np.exp(2*k*z)
        else:
            dummy = w[None,:]**3 * Sw[None,:] * np.exp(2*k[None,:]*z[:,None])
        return np.trapz(fac*dummy, w)

    # }}}


    # compute buoy heading {{{
    def compute_heading(self):
        """Compute the heading from different sources. Return heading in deg."""

        # TODO: when magnetometre will be available choose it as default
        if hasattr(self, "mag"):
            pass

        # heading signature
        heading_sig = (self.sig["heading"]/100) % 360
        
        # heading maximet
        maximet_angle = self.metadata["sensors"]["maximet"]["maximet_angle"]
        true_wnd, rel_wnd = self.met["true_wind_dir"], self.met["relative_wind_dir"]
        heading_met = (true_wnd - rel_wnd + maximet_angle) % 360

        # the low frequency heading means the angle between new BOMM y-axis and
        # true north. Magnetic deviation is taken from GPS measurements. All in
        # degrees, The mag deviation or declination is added to the current
        # magnetic mesurement ---> (check this, im not pretty sure)
        if np.isnan(heading_sig).all():
            heading = heading_met - self.gps["mag_var"][0] * 0
        else:
            heading = heading_sig - self.gps["mag_var"][0] * 0

        return heading % 360
    # }}}

    # motion matrices {{{
    def motion_matrices(self):
        """Matrices of the accelerometer, gyroscope and euler angles"""

        # check for anomalous data
        for k, v in self.ekx.items():
            if k not in ["time"]:
                self.ekx[k][abs(v) > 1E5] = np.nan

        # fill nans when possible
        for k, v in self.ekx.items():
            if k not in ["time"]:
                number_of_nans = np.isnan(v).sum()
                if (number_of_nans / len(v)) < 0.1:
                    self.ekx[k][np.isnan(v)] = np.nanmean(v)

        # construct accelerometer and gyroscope tuples
        # apply a rotation to an ENU frame of reference
        R = (np.pi, 0, np.pi/2)
        self.Acc = motcor.vector_rotation((self.ekx["accel_x"],
            self.ekx["accel_y"], self.ekx["accel_z"]), R)
        #
        self.Gyr = motcor.vector_rotation((self.ekx["gyro_x"],
            self.ekx["gyro_y"],  self.ekx["gyro_z"]),  R)

        # integrate accel and gyro to obtain euler angles
        ax, ay, ax = self.Acc
        gx, gy, gz = self.Gyr
        phi, theta = motcor.pitch_and_roll(
                ax, ay, ax, gx, gy, ax, fs=100, fc=0.04, fm=1)
        #
        # compute bomm heading and the merge with ekinox
        heading = np.radians((90 - self.compute_heading()) % 360)
        psi = motcor.yaw_from_magnetometer(self.Gyr[2], heading,
                fs=100, fc=0.04, fm=0.04)
        #
        # TODO: from BOMM3 the eknox was updated to output euler angles
        #       so we need choose phi and theta directly and psi from the
        #       combination between the magnetometre and the ekinox
        self.Eul = (phi, theta, psi)
    # }}}

    # get directional spectrum {{{
    def get_directional_spectrum(self):
        """Compute directional wave spectrum and other wave parameters."""

        # check for nans: if both are valid do nothing, else raise exception
        if (self._check_nans(self.ekx) and self._check_nans(self.wav)):
            pass
        else:
            raise Exception("Number of nans is more than 30%")

        # # if heading is greater than 90 degress return an error
        # std_yaw = np.nanstd(self.Eul[2]) * 180/np.pi
        # if std_yaw > 15:
            # raise Exception(f"BOMM has veered too much: {std_yaw:.2f} deg")

        # TODO: check wavestaff standar deviation

        # get motion matrices
        self.motion_matrices()

        # check waestaffs in use
        valid_wires = self.metadata["sensors"]["wstaff"]["valid_wires"]
        valid_wires_index = [w - 1 for w in valid_wires]

        # check dimensions
        nfft = 1024
        npoint = 6
        ntime = len(self.wav["time"])

        # determinte position of the wavestaffs
        # TODO: since offset depends on each bomm, they should be passed
        #       as an input argument.
        x_offset, y_offset, z_offset = -0.339, 0.413, 4.45
        xx, yy = wdm.reg_array(N=5, R=0.866, theta_0=180)
        # xx, yy = xx + x_offset, yy + y_offset

        # get the sampling frequency and the resampling factor
        fs = self.metadata["sensors"]["wstaff"]["sampling_frequency"]
        q = int(100/fs)

        # allocate variables
        S = np.zeros((int(nfft/2+1), npoint))
        X, Y, Z = (np.zeros((ntime, npoint)) for _ in range(3))
        #
        # apply the correction to the surface elevation and compute fourier spc
        for i, (x, y), in enumerate(zip(xx, yy)):
            #
            # get suface elevation at each point
            z = self.wav["ws%d" % (i+1)] * 3.5/4095 + z_offset
            #
            # apply motion correction
            # fs, q = 20, 5 # BOMM1
            X[:,i], Y[:,i], Z[:,i] = motcor.position_correction((x,y,z),
                    self.Acc, self.Eul, fs=fs, fc=0.04, q=q)
            #
            # compute fourier spectrum
            # TODO: compute spectrum with homemade pwelch, it will allow to
            # discard the blocks containing nan data.
            ffrq, S[:,i] = self.welch(Z[:,i], fs=fs, nfft=nfft, overlap=int(nfft/4))

        # limit to the half of the nfft to remove high frequency noise
        S = np.mean(S[1:int(nfft/4)+1,valid_wires_index], axis=1)
        ffrq = ffrq[1:int(nfft/4)+1]

        # compute directional wave spectrum
        # TODO: create a anti-aliasing filter to decimate the time series
        dfac = 2
        d = lambda x: x[::dfac,1:]
        wfrq, dirs, E, D = wdm.fdir_spectrum(d(Z), d(X), d(Y), fs=int(fs/dfac),
                omin=-4, omax=-1, nvoice=16, ws=(30, 4))
        
        # save data in the output dictionary
        list_of_variables = {
                "ffrq" : "fourier_frequencies",
                "S"    : "frequency_spectrum",
                "wfrq" : "wavelet_frequencies",
                "dirs" : "wavelet_directions",
                "E"    : "directional_wave_spectrum",
                }
        #
        for k, v in list_of_variables.items():
            self.results[k] = eval(k)
    # }}}

    # get wave parameters {{{
    def get_wave_parameters(self):
        """Compute wave paramemters for a given directional wave spectrum."""
        
        # get data from results dictionary
        # TODO check if self.get_directional_spectrum was called
        E = self.results["E"]
        S = self.results["S"]
        ffrq = self.results["ffrq"]
        wfrq = self.results["wfrq"]
        dirs = self.results["dirs"]

        # compute bulk wave parameters and stokes drift magnitude
        Hm0, Tp, pDir, mDir = self.wave_parameters(wfrq, dirs, E)
        Us0 = self.stokes_drift(ffrq, S, z=0.0)

        # save data in the output dictionary
        list_of_variables = {
                "Hm0"  : "significant_wave_height",
                "Tp"   : "peak_wave_period",
                "pDir" : "peak_wave_direction",
                "mDir" : "average_wave_direction",
                "Us0"  : "surface_stokes_drift"
                }
        #
        for k, v in list_of_variables.items():
            self.results[k] = eval(k)
    # }}}

    # write directional specrtum {{{
    def write_directional_spectrum(self):
        """Write directional spectrum to a file in the disk"""

        E = np.uint16(self.results["E"][::5,:] * 1E4)
        np.savetxt("test.spectrum.txt", E, fmt='%4d')
        np.save("test.spectrum", E)
        np.savez_compressed("test.spectrum", E)
        
    # }}}



if __name__ == "__main__":

    # define the path of the metadata file
    metafile = "../metadata/bomm1_its.yml"
    
    # define the specific date
    date = dt.datetime(2017,11,17,0,0)

    # create instance of the main class
    p = ProcessingData(metafile, date)

    # load the data, perform motion_correction and get_spectrum
    p.read_data()
    p.get_directional_spectrum()
    p.write_directional_spectrum()

    # [optional] get the wave parameters
    p.get_wave_parameters()
