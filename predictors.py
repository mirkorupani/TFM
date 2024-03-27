import json
from auxFunc import getTempExt, getCoord
import calendar
import xarray as xr
import copernicusmarine
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd

class Predictors():
    """Class to get and load the predictors for the model"""

    # MeteoGalicia Thredds URL
    threddsURLMG = "https://mandeo.meteogalicia.es/thredds/dodsC/modelos/WRF_HIST/d02/"


    def __init__(self, configFilePath, hisFile=None, folder="predictors"):

        with open(configFilePath) as f:
            self.config = json.load(f)
        
        if hisFile is None:
            folder = self.config["predictors"]["predictorsFolder"]
            self.tempExt = getTempExt(self.config["predictands"]["hisFile"])
            self.coord = getCoord(self.config["predictands"]["hisFile"], self.config["predictands"]["station"])
            
        else:
            self.tempExt = getTempExt(hisFile)
            self.coord = getCoord(hisFile, self.config["predictands"]["station"])
    
        self.windData = self.getWindData(folder=folder)
        self.hydroData = self.getHydroData(folder=folder)
        self.dischargeData = self.getDischargeData(folder=folder)
    

    def getWindURLs(self):
        """Gets a list of URLs to download the wind data from MeteoGalicia
        :return: list, URLs to download the wind data"""

        # Expand the temporal extension one hour
        iniDate = self.tempExt[0] - pd.Timedelta(hours=1)
        endDate = self.tempExt[1] + pd.Timedelta(hours=1)

        # Get the files
        urls = []
        for year in range(iniDate.year, endDate.year+1):
            for month in range(1, 13):
                if year == iniDate.year and month < iniDate.month:
                    continue
                if year == endDate.year and month > endDate.month:
                    continue
                # Get number of days in the month
                days = calendar.monthrange(year, month)[1]
                for day in range(1, days+1):
                    if year == iniDate.year and month == iniDate.month and day < iniDate.day:
                        continue
                    if year == endDate.year and month == endDate.month and day > endDate.day:
                        continue
                    urls.append(f"{self.threddsURLMG}{year}/{month:02d}/wrf_arw_det_history_d02_{year}{month:02d}{day:02d}_0000.nc4")
                    if (year > 2013) or (year == 2013 and month > 1) or (year == 2013 and month == 1 and day > 27):
                        urls.append(f"{self.threddsURLMG}{year}/{month:02d}/wrf_arw_det_history_d02_{year}{month:02d}{day:02d}_1200.nc4")
        
        return urls


    def getWindData(self, writeNetCDF=True, overwrite=False, provider="meteogalicia", folder="predictors"):
        """Gets the wind data
        :param writeNetCDF: bool, whether to write the wind data to a netCDF file
        :return: xarray.Dataset, wind data"""

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder,"windData.nc")):
            return xr.open_dataset(os.path.join(folder,"windData.nc"))

        if provider == "meteogalicia":

            windUrls = self.getWindURLs()

            # Define Lambert Conformal Conic projection
            lcc_proj = ccrs.LambertConformal(
                standard_parallels=(43.0, 43.0),
                central_longitude=345.8999938964844,
                central_latitude=34.823001861572266,
                false_easting=536.40234 * 1000,
                false_northing=-18.55861 * 1000
            )

            # Get x and y coordinates
            x, y = lcc_proj.transform_point(self.coord[1], self.coord[0], ccrs.PlateCarree())
            # Convert to km
            x /= 1000
            y /= 1000

            # Create multi-file dataset
            try:
                with xr.open_mfdataset(windUrls, decode_times=True) as ds:
                    # Select variables of interest
                    ds = ds[["u", "v", "mslp"]]
                    # Select the closest value to the point of interest
                    ds = ds.sel(x=x, y=y, method="nearest")
                
                    # Write to netCDF
                    if writeNetCDF:
                        ds.to_netcdf(os.path.join(folder,"windData.nc"))

            except ValueError:
                datasets = []
                for url in windUrls:
                    with xr.open_dataset(url, decode_times=True) as ds:
                        # Select variables of interest
                        ds = ds[["u", "v", "mslp"]]
                        # Select the closest value to the point of interest
                        ds = ds.sel(x=x, y=y, method="nearest")
                        datasets.append(ds)
                
                ds = xr.concat(datasets, dim="time")

                # If there are repeated times, keep only the last one
                ds = ds.drop_duplicates("time", keep="last")

                # Trim time to the temporal extension
                ds = ds.sel(time=slice(self.tempExt[0], self.tempExt[1]))
                    
                # Write to netCDF
                if writeNetCDF:
                    ds.to_netcdf(os.path.join(folder,"windData.nc"))
    
            return ds
        
        else:
            raise ValueError("Wind data source not recognized")
    

    def getHydroData(self, writeNetCDF=True, overwrite=False, folder="predictors"):
        """Downloads the hydrodynamic data
        :param writeNetCDF: bool, whether to write the hydrodynamic data to a netCDF file
        :return: xarray.Dataset, hydrodynamic data"""

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder, "hydroData.nc")):
            return xr.open_dataset(os.path.join(folder, "hydroData.nc"))

        # Check if copernicusmarine-credentials already exists
        if not os.path.exists(os.path.join(os.path.expanduser("~"), ".copernicusmarine", ".copernicusmarine-credentials")):
            copernicusmarine.login()

        try:
            ds = copernicusmarine.open_dataset(
                dataset_id=self.config["predictors"]["hydro"]["dataset_id"],
                maximum_longitude=self.config["predictors"]["hydro"]["point"][1],
                minimum_longitude=self.config["predictors"]["hydro"]["point"][1],
                maximum_latitude=self.config["predictors"]["hydro"]["point"][0],
                minimum_latitude=self.config["predictors"]["hydro"]["point"][0],
                start_datetime=self.tempExt[0],
                end_datetime=self.tempExt[1],
                variables=self.config["predictors"]["hydro"]["variables"]

            )
        except:
            raise ValueError("Error downloading the hydrodynamic data from Copernicus Marine")
        
        # Remove latitude and longitude dimensions
        for var in ds.data_vars:
            if "latitude" in ds[var].dims:
                ds[var] = ds[var].isel(latitude=0)
            if "longitude" in ds[var].dims:
                ds[var] = ds[var].isel(longitude=0)

        # Check for NaN values
        all_nan = True
        for var in ds.data_vars:
            if not np.isnan(ds[var].values).all():
                all_nan = False
                break
        if all_nan:
            raise ValueError("All the hydrodynamic data is NaN. Select a different point.")
        
        # Write to netCDF
        if writeNetCDF:
            ds.to_netcdf(os.path.join(folder, "hydroData.nc"))
    
        return ds
    

    def getDischargeData(self, writeNetCDF=True, overwrite=False, folder="predictors"):
        pass

