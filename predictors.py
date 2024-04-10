import json
from auxFunc import getTempExt, getCoord, datenumToDatetime
import calendar
import xarray as xr
import copernicusmarine
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import scipy.io as sio
from scipy.signal import find_peaks

class Predictors():
    """Class to get and load the predictors for the model"""

    # MeteoGalicia Thredds URL
    threddsURLMG = "https://mandeo.meteogalicia.es/thredds/dodsC/modelos/WRF_HIST/d02/"


    def __init__(self, configFilePath, hisFile=None, folder="predictors"):

        with open(configFilePath) as f:
            self.config = json.load(f)
        
        if hisFile is None:
            folder = self.config["predictors"]["predictorsFolder"]
            if isinstance(self.config["predictands"]["hisFile"], list):
                self.tempExt = list()
                for hisFile in self.config["predictands"]["hisFile"]:
                    self.tempExt.append(getTempExt(hisFile))
                self.coord = getCoord(self.config["predictands"]["hisFile"][0], self.config["predictands"]["station"])
            else:
                self.tempExt = getTempExt(self.config["predictands"]["hisFile"])
                self.coord = getCoord(self.config["predictands"]["hisFile"], self.config["predictands"]["station"])
            
        else:
            self.tempExt = getTempExt(hisFile)
            self.coord = getCoord(hisFile, self.config["predictands"]["station"])
    
        self.windData = self.getWindData(folder=folder)
        self.hydroData = self.getHydroData(folder=folder)
        self.dischargeData = self.getDischargeData(folder=folder)
        self.tidalRangeData = self.getTidalRangeData(folder=folder)
    

    def getWindURLs(self):
        """Gets a list of URLs to download the wind data from MeteoGalicia
        :return: list, URLs to download the wind data"""

        # Expand the temporal extension one hour
        if isinstance(self.tempExt, list):
            iniDate = list()
            endDate = list()
            for i in range(len(self.tempExt)):
                iniDate.append(self.tempExt[i][0] - pd.Timedelta(hours=1))
                endDate.append(self.tempExt[i][1] + pd.Timedelta(hours=1))
        else:
            iniDate = self.tempExt[0] - pd.Timedelta(hours=1)
            endDate = self.tempExt[1] + pd.Timedelta(hours=1)

        # Get the files
        urls = []
        if isinstance(iniDate, list):
            for i in range(len(iniDate)):
                for year in range(iniDate[i].year, endDate[i].year+1):
                    for month in range(1, 13):
                        if year == iniDate[i].year and month < iniDate[i].month:
                            continue
                        if year == endDate[i].year and month > endDate[i].month:
                            continue
                        # Get number of days in the month
                        days = calendar.monthrange(year, month)[1]
                        for day in range(1, days+1):
                            if year == iniDate[i].year and month == iniDate[i].month and day < iniDate[i].day:
                                continue
                            if year == endDate[i].year and month == endDate[i].month and day > endDate[i].day:
                                continue
                            urls.append(f"{self.threddsURLMG}{year}/{month:02d}/wrf_arw_det_history_d02_{year}{month:02d}{day:02d}_0000.nc4")
                            if (year > 2013) or (year == 2013 and month > 1) or (year == 2013 and month == 1 and day > 27):
                                if year == 2013 and month == 2 and day == 23:
                                    continue
                                urls.append(f"{self.threddsURLMG}{year}/{month:02d}/wrf_arw_det_history_d02_{year}{month:02d}{day:02d}_1200.nc4")
        else:
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
                            if year == 2013 and month == 2 and day == 23:
                                    continue
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
                if isinstance(self.tempExt, list):
                    dsList = []
                    for i in range(len(self.tempExt)):
                        dsList.append(ds.sel(time=slice(self.tempExt[i][0], self.tempExt[i][1])))
                    ds = xr.concat(dsList, dim="time")
                else:
                    ds = ds.sel(time=slice(self.tempExt[0], self.tempExt[1]))
                    
                # Write to netCDF
                if writeNetCDF:
                    try:
                        ds.to_netcdf(os.path.join(folder, "windData.nc"))
                    except:
                       # Save each variable separately
                        for var_name in ds.data_vars:
                            ds[var_name].to_netcdf(os.path.join(folder, f'{var_name}.nc'))
                        # Merge the variables
                        ds = xr.merge([xr.open_dataset(os.path.join(folder, f'{var_name}.nc')) for var_name in ds.data_vars])
                        ds.to_netcdf(os.path.join(folder, "windData.nc"))
                        # Remove temporary files
                        # for var_name in ds.data_vars:
                        #     os.remove(os.path.join(folder, f'{var_name}.nc'))
    
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
            # Check if temporal extension is a list
            if isinstance(self.tempExt, list):
                dsList = []
                for i in range(len(self.tempExt)):
                    ds = copernicusmarine.open_dataset(
                        dataset_id=self.config["predictors"]["hydro"]["dataset_id"],
                        maximum_longitude=self.config["predictors"]["hydro"]["point"][1],
                        minimum_longitude=self.config["predictors"]["hydro"]["point"][1],
                        maximum_latitude=self.config["predictors"]["hydro"]["point"][0],
                        minimum_latitude=self.config["predictors"]["hydro"]["point"][0],
                        start_datetime=self.tempExt[i][0],
                        end_datetime=self.tempExt[i][1],
                        variables=self.config["predictors"]["hydro"]["variables"]
                    )
                    dsList.append(ds)
                ds = xr.concat(dsList, dim="time")
                ds = ds.drop_duplicates("time", keep="first")
            else:
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


    def getTidalRangeData(self, writeNetCDF=True, overwrite=False, folder="predictors"):

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder, "tidalRangeData.nc")):
            return xr.open_dataset(os.path.join(folder, "tidalRangeData.nc"))
                                   
        # Import tide gauge mat file
        data = sio.loadmat(self.config["predictors"]["tidalRange"])
        time = np.array(datenumToDatetime(data["Date"]))
        astronomicTide = data["Ma"]
        # Trim time to the temporal extension
        if isinstance(self.tempExt, list):
            idx = np.where((time >= self.tempExt[0][0].to_pydatetime()) & (time <= self.tempExt[-1][1].to_pydatetime()))[0]
        else:
            idx = np.where((time >= self.tempExt[0].to_pydatetime()) & (time <= self.tempExt[1].to_pydatetime()))[0]
        time = time[idx]
        astronomicTide = astronomicTide[idx]
        peaks, _ = find_peaks(astronomicTide.flatten())
        valleys, _ = find_peaks(-astronomicTide.flatten())
        # Concatenate peaks and valleys
        peaks = np.sort(np.concatenate((peaks, valleys)))

        # Calculate tidal range
        tidalRange = np.empty(len(astronomicTide))
        for i in range(len(tidalRange)):
            if i in peaks:
                tidalRange[i] = np.nan
                continue
            beforePeak = peaks[np.where(peaks < i)]
            afterPeak = peaks[np.where(peaks > i)]
            if len(beforePeak) == 0 or len(afterPeak) == 0:
                tidalRange[i] = np.nan
            else:
                tidalRange[i] = astronomicTide[afterPeak[0]] - astronomicTide[beforePeak[-1]]
        # Make all values positive
        tidalRange = np.abs(tidalRange)
        # Assign closest non-nan value to nan values
        nanIdx = np.where(np.isnan(tidalRange))[0]
        notNanIdx = np.where(~np.isnan(tidalRange))[0]
        for i in nanIdx:
            tidalRange[i] = tidalRange[notNanIdx[np.argmin(np.abs(i - notNanIdx))]]
        
        # Create xarray.Dataset
        ds = xr.Dataset(
            {
                "tidalRange": (("time",), tidalRange)
            },
            coords={"time": pd.to_datetime(time)},
        )

        # Write to netCDF
        if writeNetCDF:
            ds.to_netcdf(os.path.join(folder, "tidalRangeData.nc"))

        return ds

