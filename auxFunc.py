import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import datetime

def getTempExt(netcdf_file):
    """
    Get the temporal extension of a Delft3D netcdf output file
    :param netcdf_file: str, path to the netcdf file
    :return: pd.Timestamp tuple, (start, end)
    """
    with xr.open_dataset(netcdf_file) as ds:
        time_ext = ds.time.values

    return pd.Timestamp(time_ext[0]), pd.Timestamp(time_ext[-1])


def getCoord(netcdf_file, point, srcProj=ccrs.epsg(32630)):
    """
    Get the coordinates of a specific point in a Delft3D netcdf history output file
    :param netcdf_file: str, path to the netcdf file
    :param point: int, index of the point
    :param proj: cartopy.crs, projection to transform the coordinates from
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
    """Converts datenum to datetime
    Args:
        datenum: datenum
        Returns:
        datetime.datetime"""
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