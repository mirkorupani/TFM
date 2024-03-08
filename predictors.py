import json
from auxFunc import getTempExt, getCoord
import calendar
import xarray as xr
import copernicusmarine
import cartopy.crs as ccrs
import numpy as np
import os

class Predictors():
    """Class to get and load the predictors for the model"""

    threddsURLMG = "https://mandeo.meteogalicia.es/thredds/dodsC/modelos/WRF_HIST/d02/"


    def __init__(self, config):
        
        with open(config) as f:
            self.config = json.load(f)
        
        self.tempExt = getTempExt(self.config["hisFile"])
        self.coord = getCoord(self.config["hisFile"], self.config["point"])
        self.windData = self.getWindData()
        self.currentData = self.getCurrentData()
        self.dischargeData = self.getDischargeData()
    

    def getWindURLs(self):
        """Gets a list of URLs to download the wind data from MeteoGalicia
        :return: list, URLs to download the wind data"""

        # Get the files
        urls = []
        for year in range(self.tempExt[0].year, self.tempExt[1].year+1):
            for month in range(1, 13):
                if year == self.tempExt[0].year and month < self.tempExt[0].month:
                    continue
                if year == self.tempExt[1].year and month > self.tempExt[1].month:
                    continue
                # Get number of days in the month
                days = calendar.monthrange(year, month)[1]
                for day in range(1, days+1):
                    urls.append(f"{self.threddsURLMG}{year}/{month:02d}/wrf_arw_det_history_d02_{year}{month:02d}{day:02d}_0000.nc4")
                    if (year > 2013) or (year == 2013 and month > 1) or (year == 2013 and month == 1 and day > 27):
                        urls.append(f"{self.threddsURLMG}{year}/{month:02d}/wrf_arw_det_history_d02_{year}{month:02d}{day:02d}_1200.nc4")
        
        return urls


    def getWindData(self, writeNetCDF=True, overwrite=False):
        """Gets the wind data
        :param writeNetCDF: bool, whether to write the wind data to a netCDF file
        :return: xarray.Dataset, wind data"""

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(self.config["predictors"]["predictorsFolder"],"windData.nc")):
            return xr.open_dataset(os.path.join(self.config["predictors"]["predictorsFolder"],"windData.nc"))

        if self.config["predictors"]["wind"] == "meteogalicia":

            windUrls = self.getWindURLs()

            ### VER: creo que no está quedando bien la latitud y longitud final del punto

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
                        ds.to_netcdf(os.path.join(self.config["predictors"]["predictorsFolder"],"windData.nc"))

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
                    
                # Write to netCDF
                if writeNetCDF:
                    ds.to_netcdf(os.path.join(self.config["predictors"]["predictorsFolder"],"windData.nc"))
    
            return ds
        
        else:
            raise ValueError("Wind data source not recognized")
    

    def getCurrentData(self, writeNetCDF=True):
        """Downloads the hydrodynamic data
        :param writeNetCDF: bool, whether to write the hydrodynamic data to a netCDF file
        :return: xarray.Dataset, hydrodynamic data"""

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
            ds.to_netcdf(os.path.join(self.config["predictors"]["predictorsFolder"], "currentData.nc"))
    
        return ds
    

    def getDischargeData(self, writeNetCDF=True):
        pass

