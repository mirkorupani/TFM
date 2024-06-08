import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import datetime

def getTempExt(netcdf_file):
    """
    Gets the temporal extension of a Delft3D netcdf output file

    :param netcdf_file: str, path to the netcdf file

    :return: pd.Timestamp tuple, (start, end)
    """
    with xr.open_dataset(netcdf_file) as ds:
        time_ext = ds.time.values

    return pd.Timestamp(time_ext[0]), pd.Timestamp(time_ext[-1])


def getCoord(netcdf_file, point, srcProj=ccrs.epsg(32630)):
    """
    Gets the coordinates of a specific point in a Delft3D netcdf history output file

    :param netcdf_file: str, path to the netcdf file
    :param point: int, index of the point
    :param srcProj: cartopy.crs, source projection

    :return: tuple, (lat, lon)
    """
    with xr.open_dataset(netcdf_file) as ds:
        x = ds.x.values[point]
        y = ds.y.values[point]

    # Transform to geographic coordinates
    proj = ccrs.PlateCarree()
    lon, lat = proj.transform_point(x, y, srcProj)

    return lat, lon


def datenumToDatetime(datenum):
    """
    Converts a MATLAB datenum to a Python datetime object
    
    :param datenum: float or np.array, MATLAB datenum
    
    :return: datetime.datetime or list, Python datetime object
    """
    try:
        iter(datenum)
        is_iter = True
    except TypeError:
        is_iter = False

    if is_iter:
        days = np.floor(datenum).astype(int)
        frac_days = datenum - days
        datetime_list = [datetime.datetime.fromordinal(int(d) - 366) + datetime.timedelta(days=int(d % 1)) for d in
                         days]
        datetime_list = [dt + datetime.timedelta(microseconds=int(frac_day * 24 * 60 * 60 * 1000000)) for dt, frac_day
                         in zip(datetime_list, frac_days)]
        
        # Round to seconds
        datetime_list = [dt.replace(microsecond=0) for dt in datetime_list]

        return datetime_list

    else:
        days = np.floor(datenum).astype(int)
        frac_days = datenum - days
        datetime_obj = datetime.datetime.fromordinal(int(days) - 366) + datetime.timedelta(days=int(frac_days % 1))
        datetime_obj += datetime.timedelta(microseconds=int(frac_days * 24 * 60 * 60 * 1000000))

        # Round to seconds
        datetime_obj = datetime_obj.replace(microsecond=0)

        return datetime_obj


def willmottSkillIndex(data, dataRecon, returnNumDen=False):
    """
    Calculates the Willmott skill index between two samples.
    
    :param data: array-like, The original sample data.
    :param dataRecon: array-like, The reconstructed sample data.
    
    :param returnNumDen: bool, Whether to return the numerator and denominator of the skill index.
    
    :return: float, The Willmott skill index.
    """

    if hasattr(data, 'to_numpy'):
        xo = data.to_numpy()
    else:
        xo = data

    if hasattr(dataRecon, 'to_numpy'):
        xs = dataRecon.to_numpy()
    else:
        xs = dataRecon

    xo = np.squeeze(xo)
    xs = np.squeeze(xs)

    a = np.sum(np.square(xs - xo), axis=0)
    b = np.sum(np.square(np.abs(xs - np.nanmean(xo, axis=0)) + np.abs(xo - np.nanmean(xo, axis=0))), axis=0)

    if returnNumDen:
        return a, b
    else:
        return 1 - a / b


def ksStatistic(data, dataRecon):
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic between two samples.

    :param data: array-like, The original sample data.
    :param dataRecon: array-like, The reconstructed sample data.

    :return: float, The KS statistic.
    """
    # Convert data to numpy arrays if they are not already
    xo = np.asarray(data).flatten()
    xs = np.asarray(dataRecon).flatten()

    # Sort the data arrays
    xoSorted = np.sort(xo)
    xsSorted = np.sort(xs)

    # Calculate the maximum absolute difference between the sorted arrays
    ksStat = np.max(np.abs(xoSorted - xsSorted))

    return ksStat


def pearsonCorrCoeff(data, dataRecon):
    """
    Calculates the Pearson correlation coefficient between two samples.

    :param data: array-like, The original sample data.
    :param dataRecon: array-like, The reconstructed sample data.

    :return: float, The Pearson correlation coefficient.
    """
    # Convert data to numpy arrays if they are not already
    xo = np.asarray(data).flatten()
    xs = np.asarray(dataRecon).flatten()

    # Calculate the Pearson correlation coefficient
    pearsonCoeff = np.corrcoef(xo, xs)[0, 1]

    return pearsonCoeff


def perkinsSkillScore(data, dataRecon, bins=50):
    """
    Calculates the Perkins skill score between two samples.

    :param data: array-like, The original sample data.
    :param dataRecon: array-like, The reconstructed sample data.
    :param bins: int, The number of bins to use for the histograms.

    :return: float, The Perkins skill score.
    """
    # Compute histograms for the observed and modeled data
    zo, bin_edges = np.histogram(data, bins=bins, density=True)
    zm, _ = np.histogram(dataRecon, bins=bin_edges, density=True)

    # Normalize the histograms to form PDFs
    zo = zo / np.sum(zo)
    zm = zm / np.sum(zm)

    # Compute the cumulative minimum value of the two distributions
    sscore = np.sum(np.minimum(zo, zm))

    return sscore


def concatCamel(strings):
    """
    Concatenates strings in lowerCamelCase format
    
    :param strings: list, list of strings
    
    :return: str, concatenated string in lowerCamelCase format
    """
    if not strings:
        return ''
    
    # Convert the first string to lowercase
    result = strings[0].lower()
    
    # Capitalize the first letter of each subsequent string
    for string in strings[1:]:
        result += string.capitalize()
    
    return result