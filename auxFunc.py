import xarray as xr
import cartopy.crs as ccrs
import pandas as pd

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