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
"""

# --- import libs ---
import numpy as np
import datetime as dt
import pandas as pd
import csv
import sys
import yaml


# === Read Data Class === 
class ReadRawData(object):

    """
    This class contains methods to read the csv files written in the SSD for the
    CICESE-BOMM (Oceanographic and Marine Meteorology Buoys).
    """

    __slots__ = ["metadata", "basepath", "bomm_name"]

    # private methods to read data {{{
    def __init__(self, metafile):
        """Function to initialize the class.
        
        Args:
            metafile (string): Name of the YAML file containg BOMM metadata.

        Usage:
            import bomm
            metafile = "bomm.yml"
            b = bomm.ReadRowData(metafile)
            wav = b.read(sensor="wstaff", date=dt.datetime(2017,11,17,0,0))

        Sensors:
            ekinox, sonic, gps, maximet, proceanus,
            rbr, signature, vector, wstaff
        
        TODO:
            - [ ] create function to read UCMR.

        """

        self.metadata = self._readmetadata(metafile)
        self.basepath = self.metadata["basepath"]
        self.bomm_name = self.metadata["name"]


    # parse hour line into floating 
    def _parsedate(self, string):
        """Parse the date HH:MM:SS.ffffff YYYY mm dd."""

        hour, dd, mm, yy = string.split()
        H, M, sec = hour.split(":")
        try:
            S, f = sec.split(".")
        except ValueError:
            S, f = sec, '0'
        return  dt.datetime(int(yy), int(mm), int(dd), int(H), int(M), int(S), int(f))

    
    # get file name from a given date
    def _getfilename(self, sensor, date):
        """Returns filename based on date, sensor, and path."""
        
        if sensor in ["maximet"]:                      # <--- one-hour files
            fmt = "/{0}/%Y/%m/%d/{0}-%y%m%d%H.csv".format(sensor)
        #
        elif sensor in ["rbr", "proceanus"]:           # <--- one-day files
            fmt = "/{0}/%Y/%m/{0}-%y%m%d.csv".format(sensor)
        #
        else:                                          # <--- one-minute files
            fmt = "/{0}/%Y/%m/%d/%H/{0}-%y%m%d%H%M.csv".format(sensor)
        
        filename = self.basepath + self.bomm_name + \
                   "/data" + dt.datetime.strftime(date, fmt)
        return filename


    # read metadada
    def _readmetadata(self, metafile):
        """Read the YAML file containg metadata."""

        with open(metafile, "r") as f:
            return yaml.load(f)


    # read sampling frequency
    def _getsampfreq(self, sensor):
        """Returns the sampling frequency for specific sensor."""

        sensors = self.metadata["sensors"][sensor]
        try:
            fs = float(sensors["sampling_frequency"])
        except ValueError as e:
            num, den = sensors["sampling_frequency"].split("/")
            fs = float(num) / float(den)

        return fs
            

    # read seconds per file
    def _getsecperfile(self, sensor):
        """Returns the seconds per file for specific sensor."""

        sensors = self.metadata["sensors"][sensor]
        return int(sensors["seconds_per_file"])
 

    # read columns
    def _getcolumns(self, sensor):
        """Returns the columns to be read for specific sensor."""

        sensors = self.metadata["sensors"][sensor]
        return {k:v["column"] for k,v in sensors["variables"].items()}

    # convert to float if you can
    def _float(self, x):
        """Convert string to float even if it has double decimal point."""

        try:
            return float(x)
        except ValueError:
            #
            # if value has double decimal
            if len(x.split(".")) >= 2:
                return np.nan
            #
            # if value is a letter
            elif x.isalpha():
                return x
            #
            # finally, return nan
            else:
                return np.nan

    # read file
    def _readfile(self, sensor, date, columns):
        """This function reads the data of the original files."""

        # get filename
        filename = self._getfilename(sensor, date)

        # pre-define variables
        time = []
        obs = {v: [] for v in columns.keys()}

        # TODO: check if file has null bytes

        # read file
        with open(filename, 'r') as f:
            data = csv.reader(f, delimiter=',')
            for irow, row in enumerate(data):
                #
                # parse date and append it to a list
                time.append(self._parsedate(" ".join(row[:4])))
                #
                # loop for each column
                for k, v in columns.items():
                    #
                    # check if data is a 2d array
                    if isinstance(v, list):
                        a, b = v
                        obs[k].append([self._float(s) for s in row[a:b+1]])
                    #
                    # if not, then it is a 1d array
                    else:
                        try:
                            #
                            # try to convert each value into float
                            obs[k].append(self._float(row[v]))
                        except IndexError:
                            #
                            # if line is empty then fill with nans
                            obs[k].append(np.nan)
                        # except ValueError:
                            # #
                            # # if conversion fails then leave as string
                            # obs[k].append(row[v])
                        except:
                            raise Exception("Problems reading data")

        # convert to numpy array and add time array
        obs["time"] = time
        for k, v in obs.items():
            obs[k] = np.asarray(v)
        
        return obs


    # new time array
    def _getnewtime(self, date, fs, N):
        """Returns list of datetimes of N seconds at fs sampling frequency."""
        
        seconds = np.arange(0, N, 1/fs)
        return np.array([date + dt.timedelta(seconds=s) for s in seconds])


    # return invalid file
    def _returninvalid(self, sensor, date):
        """Returns NAN with the same structure as the valid data."""

        N = self._getsecperfile(sensor)
        fs = self._getsampfreq(sensor)
        columns = self._getcolumns(sensor)

        data = {"time": self._getnewtime(date, fs, N)}
        for k, v in columns.items():
            if isinstance(v, list):
                data[k] = np.full((len(data["time"]), v[1]-v[0]+1), np.nan)
            else:
                data[k] = np.full(len(data["time"]), np.nan)

        return data


    # interpolate the data
    # TODO: I need to improve this code function
    def _resample(self, dic, date, fs, N=600):
        """This function uses pandas for an accurate resample"""

        # check minimum sampling frequency
        # fs = np.max((fs, 1./600.)) # force to be one data each ten minutes 

        # create new time array
        time_new = self._getnewtime(date, fs, N)

        # create pandas dataframe using the data in the dictionary
        t = dic["time"]
        df = pd.DataFrame({k:v for k,v in dic.items() 
                               if k not in ["time"]}, index=t)

        # check time discontinuities
        max_gap = np.diff(dic["time"]).max().total_seconds()
        if max_gap > 5 / fs:
            pass

        # check the number of nans
        n = len(t)
        n_nans = df.isnull().sum().max()
        if n_nans / n > 0.1:
            raise Exception("Number of NaNs is %d out of %d." % (n_nans, n))

        # drop missing values only if are less than 10 percent of the data
        # TODO: limit to a number of consecutive nans of one second
        df = df.dropna(how="any")

        # remove duplicate indices if they exist
        df = df[~df.index.duplicated(keep='first')]
        
        # sort data in ascending and reindex to the new time
        # perform the reindex twice, one backward and one foreward
        # limit to a number of consecutive nans of one second
        # l = int(fs) if fs>=1 else 1
        l = 1
        df = df.sort_index().reindex(time_new, limit=l, method="bfill").ffill()

        # crate new dictionary for output
        outdic = {c:df[c].values for c in df}
        outdic["time"] = time_new
        return outdic


    # remove NULL bytes
    def _remove_nullbytes(self, filename):
        """Remove NULL bytes in file represented by '\x00'."""

        with open(filename, "r") as f:
            data = f.read()

            if data.find("\x00") != -1:
                f.write(data.replace("\x00", ""))
            
            return f
    # }}}

    # read ekinox {{{
    def ekinox(self, date):

        # parameters
        fs = self._getsampfreq("ekinox")
        N = self._getsecperfile("ekinox")
        columns = self._getcolumns("ekinox")

        # load observed data
        obs = self._readfile("acelerometro", date, columns)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read sonic {{{
    def sonic(self, date):

        # parameters
        fs = self._getsampfreq("sonic")
        N = self._getsecperfile("sonic")
        columns = self._getcolumns("sonic")

        # load observed data
        obs = self._readfile("anemometro", date, columns)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read gps {{{
    def gps(self, date):
        
        # parameters
        fs = self._getsampfreq("gps")
        N = self._getsecperfile("gps")
        columns = self._getcolumns("gps")

        # load observed data
        obs = self._readfile("gps", date, columns)

        # assign sign to latitud or longitude
        sign = np.vectorize(lambda s: -1 if s in ["S", "W"] else 1)
        obs["latitude"] *= sign(obs["lat_sign"])
        obs["longitude"] *= sign(obs["lon_sign"])

        # remove items
        for a in ["status","lon_sign", "lat_sign"]:
            obs.pop(a)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read marvi {{{
    def marvi(self, date):

        # parameters
        fs = self._getsampfreq("marvi")
        N = self._getsecperfile("marvi")
        columns = self._getcolumns("marvi")

        # load observed data
        obs = self._readfile("marvi", date, columns)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read maximet {{{
    def maximet(self, date):

        # parameters
        date = date.replace(minute=0)
        fs = self._getsampfreq("maximet")
        N = self._getsecperfile("maximet")
        columns = self._getcolumns("maximet")

        # load observed data
        obs = self._readfile("maximet", date, columns)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read proceanus {{{
    def proceanus(self, date):

        # parameters
        date = date.replace(hour=0, minute=0)
        fs = self._getsampfreq("proceanus")
        N = self._getsecperfile("proceanus")
        columns = self._getcolumns("proceanus")

        # load observed data
        obs = self._readfile("proceanus", date, columns)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read rbr {{{
    def rbr(self, date):

        # parameters
        date = date.replace(hour=0, minute=0)
        fs = self._getsampfreq("rbr")
        N = self._getsecperfile("rbr")
        columns = self._getcolumns("rbr")

        # load observed data
        obs = self._readfile("rbr", date, columns)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read signature {{{
    def signature(self, date):

        # parameters
        fs = self._getsampfreq("signature")
        N = self._getsecperfile("signature")
        columns = self._getcolumns("signature")

        # load observed data
        obs = self._readfile("signature", date, columns)

        # get active beams and number of cells
        beams = self.metadata["sensors"]["signature"]["beams"]
        ncell = self.metadata["sensors"]["signature"]["ncell"]

        # resample velocity, amplitude and correlation
        for beam in beams:
            for letter in ["vel", "amp", "cor"]:
                variable = "{0}_b{1}".format(letter, beam)
                arr = obs[variable]
                for i in range(ncell):
                    obs["%s_c%02d" % (variable, i+1)] = arr[:,i]
                #
                # remove arr
                obs.pop("%s" % (variable))

        # resample data
        dic = self._resample(obs, date, fs, N)

        # return to two-dimensional arrays
        for beam in beams:
            for letter in ["vel", "amp", "cor"]:
                variable = "{0}_b{1}".format(letter, beam)
                arr = np.full((len(dic["time"]), ncell), np.nan)
                for i in range(ncell):
                    arr[:,i] = dic["%s_c%02d" % (variable, i+1)]
                    dic.pop("%s_c%02d" % (variable, i+1))
                #
                dic[variable] = arr

        return dic

    # }}}

    # read vector {{{
    def vector(self, date):

        # parameters
        fs = self._getsampfreq("vector")
        N = self._getsecperfile("vector")
        columns = self._getcolumns("vector")

        # load observed data
        obs = self._readfile("vector", date, columns)

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read wavestaff {{{
    def wstaff(self, date):

        # parameters
        fs = self._getsampfreq("wstaff")
        N = self._getsecperfile("wstaff")
        columns = self._getcolumns("wstaff")

        # load observed data
        obs = self._readfile("wstaff", date, columns)

        # apply correction factor of 3.5/4095
        # for k in columns.values():
            # obs[k] *= 3.5/4095 

        # return data
        return self._resample(obs, date, fs, N)
    # }}}

    # read any sensor {{{
    def read(self, sensor, date, logfile=sys.stderr):
        """Wrapper function to handle errors when reading data for each sensor."""

        function = eval("self.%s" % sensor)
        try:
            # read data
            return function(date)

        # if file does not exist or not valid
        except Exception as e:
            error = "%10s ---> {%s} : %s" % (sensor, date, e)
            print(error, file=logfile)
            return self._returninvalid(sensor, date)
    # }}}



if __name__ == "__main__":

    pass

# === end of file ===


