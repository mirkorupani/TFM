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
from datetime import timedelta
import requests
import xml.etree.ElementTree as ET
from utide import solve, reconstruct


class Predictors():
    """Class to get and load the predictors for the model"""

    # MeteoGalicia Thredds URL
    threddsURLMG = "https://mandeo.meteogalicia.es/thredds/dodsC/modelos/WRF_HIST/d02/"


    def __init__(self, config, hisFile=None, folder="predictors"):
        """
        Initializes the Predictors class
        
        :param config: str, path to the configuration file or dictionary with the configuration
        :param hisFile: str, path to the netCDF HIS file
        :param folder: str, folder where the predictors are stored
        
        :return: None
        """

        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
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
    
        self.windData = self.getWindData(folder=folder, batchSize=5)
        self.hydroData = self.getHydroData(folder=folder)
        self.dischargeData = self.getDischargeData(folder=folder)
        self.tidalRangeData = self.getTidalRangeData(folder=folder)


    def processBatch(self, urls, x, y, batchNumber, write_netcdf=False, folder="predictors"):
        """
        Processes a batch of wind data files
        
        :param urls: list, URLs of the wind data files
        :param x: float, x coordinate of the point of interest
        :param y: float, y coordinate of the point of interest
        :param batchNumber: int, batch number
        :param write_netcdf: bool, whether to write the batch to a netCDF file
        :param folder: str, folder where the batch is stored
        
        :return: None
        """

        with xr.open_mfdataset(urls, decode_times=True, chunks="auto", combine="nested", combine_attrs="override", compat="override") as ds:
            # Select variables of interest
            ds = ds[["u", "v", "mslp"]]
            
            # Select the closest value to the point of interest
            ds = ds.sel(x=x, y=y, method="nearest")
            
        if write_netcdf:
            # If '_NCProperties' attribute is present, remove it
            if '_NCProperties' in ds.attrs:
                del ds.attrs['_NCProperties']
            # Write to NetCDF
            ds.to_netcdf(os.path.join(folder, f"windData{batchNumber}.nc"))
    

    def processInBatches(self, file_urls, x, y, batchSize=100, write_netcdf=False, folder="predictors"):
        """
        Processes the wind data in batches
        
        :param file_urls: list, URLs of the wind data files
        :param x: float, x coordinate of the point of interest
        :param y: float, y coordinate of the point of interest
        :param batchSize: int, size of the batch
        :param write_netcdf: bool, whether to write the wind data to a netCDF file
        :param folder: str, folder where the wind data is stored
        
        :return: xarray.Dataset, wind data
        """

        for batch, i in enumerate(range(0, len(file_urls), batchSize)):
            # Check if batch netCDF file exists
            if os.path.exists(os.path.join(folder, f"windData{batch+1}.nc")):
                continue
            batchUrls = file_urls[i:i+batchSize]
            self.processBatch(batchUrls, x, y, batch + 1, write_netcdf, folder=folder)
        
        # Open the netCDF files
        ds = xr.open_mfdataset([os.path.join(folder, f"windData{batch}.nc") for batch in range(1, int(len(file_urls)/batchSize)+1)], combine="nested", concat_dim="time")

        ds = self.processDS(ds)

        # Save the dataset to a single netCDF file
        ds.to_netcdf(os.path.join(folder, "windData.nc"))

        return ds


    def getWindData(self, writeNetCDF=True, overwrite=False, provider="meteogalicia", folder="predictors", batchSize=100):
        """
        Downloads the wind data
        
        :param writeNetCDF: bool, whether to write the wind data to a netCDF file
        :param overwrite: bool, whether to overwrite the wind data file
        :param provider: str, wind data provider
        :param folder: str, folder where the wind data is stored
        :param batchSize: int, size of the batch
        
        :return: xarray.Dataset, wind data
        """

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder, "windData.nc")):
            return xr.open_dataset(os.path.join(folder, "windData.nc"))
        
        if provider == "meteogalicia":
            
            baseCatalogURL = "https://mandeo.meteogalicia.es/thredds/catalog/modelos/WRF_HIST/d02/"
            baseURL = "https://mandeo.meteogalicia.es/thredds/dodsC/"

            windUrls = self.getThreddsURLs(baseCatalogURL, baseURL)

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
            # try:
            return self.processInBatches(windUrls, x, y, write_netcdf=writeNetCDF, batchSize=batchSize, folder=folder)
        
        else:
            raise ValueError("Wind data source not recognized")


    def processDS(self, ds):
        """
        Processes the wind data
        
        :param ds: xarray.Dataset, wind data
        
        :return: xarray.Dataset, processed wind data
        """

        ds = ds.drop_duplicates("time", keep="last")
        
        # Trim time to the temporal extension
        if isinstance(self.tempExt, list):
            dsList = []
            for i in range(len(self.tempExt)):
                dsList.append(ds.sel(time=slice(self.tempExt[i][0], self.tempExt[i][1])))
            ds = xr.concat(dsList, dim="time")

            ds = ds.drop_duplicates("time", keep="last")

        else:
            ds = ds.sel(time=slice(self.tempExt[0], self.tempExt[1]))

        return ds
    

    def getHydroData(self, writeNetCDF=True, overwrite=False, folder="predictors"):
        """
        Downloads the hydrodynamic data
        
        :param writeNetCDF: bool, whether to write the hydrodynamic data to a netCDF file
        :param overwrite: bool, whether to overwrite the hydrodynamic data file
        :param folder: str, folder where the hydrodynamic data is stored
        
        :return: xarray.Dataset, hydrodynamic data
        """

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder, "hydroData.nc")):
            return xr.open_dataset(os.path.join(folder, "hydroData.nc"))

        # Check if copernicusmarine-credentials already exists
        if not os.path.exists(os.path.join(os.path.expanduser("~"), ".copernicusmarine", ".copernicusmarine-credentials")):
            copernicusmarine.login()

        try:
            dsList = dict()
            # Check if temporal extension is a list
            if isinstance(self.tempExt, list):
                for dataset_id, variables in zip(self.config["predictors"]["hydro"]["dataset_id"], self.config["predictors"]["hydro"]["variables"]):
                    dsList[dataset_id] = list()
                    for i in range(len(self.tempExt)):

                        hydroDataset = copernicusmarine.open_dataset(
                            dataset_id=dataset_id,
                            maximum_longitude=self.config["predictors"]["hydro"]["point"][1],
                            minimum_longitude=self.config["predictors"]["hydro"]["point"][1],
                            maximum_latitude=self.config["predictors"]["hydro"]["point"][0],
                            minimum_latitude=self.config["predictors"]["hydro"]["point"][0],
                            start_datetime=self.tempExt[i][0],
                            end_datetime=self.tempExt[i][1],
                            variables=variables
                        )

                        # If there is a depth coordinate, keep only superficial values
                        if "depth" in hydroDataset.coords:
                            hydroDataset = hydroDataset.isel(depth=0)

                        # If the inferred frequency is not hourly, resample to hourly
                        if hydroDataset.time.to_index().inferred_freq != "h":
                            hydroDataset = hydroDataset.resample(time="H").interpolate("linear")
                            print(f"Resampled data to hourly frequency for dataset {dataset_id} and temporal extension {self.tempExt[i][0]} - {self.tempExt[i][1]}")

                        dsList[dataset_id].append(hydroDataset)
                    
                    dsList[dataset_id] = xr.concat(dsList[dataset_id], dim="time")
                    dsList[dataset_id] = dsList[dataset_id].drop_duplicates("time", keep="first")

                # Merge datasets
                hydroDataset = xr.merge(dsList.values())

            else:
                for dataset_id, variables in zip(self.config["predictors"]["hydro"]["dataset_id"], self.config["predictors"]["hydro"]["variables"]):

                    hydroDataset = copernicusmarine.open_dataset(
                        dataset_id=dataset_id,
                        maximum_longitude=self.config["predictors"]["hydro"]["point"][1],
                        minimum_longitude=self.config["predictors"]["hydro"]["point"][1],
                        maximum_latitude=self.config["predictors"]["hydro"]["point"][0],
                        minimum_latitude=self.config["predictors"]["hydro"]["point"][0],
                        start_datetime=self.tempExt[0],
                        end_datetime=self.tempExt[1],
                        variables=variables
                    )

                    # If there is a depth coordinate, keep only superficial values
                    if "depth" in hydroDataset.coords:
                        hydroDataset = hydroDataset.isel(depth=0)

                    # If the inferred frequency is not hourly, resample to hourly
                    if hydroDataset.time.to_index().inferred_freq != "h":
                        hydroDataset = hydroDataset.resample(time="H").interpolate("linear")
                        print(f"Resampled data to hourly frequency for dataset {dataset_id} and temporal extension {self.tempExt[0]} - {self.tempExt[1]}")

                    dsList[dataset_id] = hydroDataset

                # Merge datasets
                hydroDataset = xr.merge(dsList.values())
                
        except:
            raise ValueError("Error downloading the hydrodynamic data from Copernicus Marine")
        
        # Remove latitude and longitude dimensions
        for var in hydroDataset.data_vars:
            if "latitude" in hydroDataset[var].dims:
                hydroDataset[var] = hydroDataset[var].isel(latitude=0)
            if "longitude" in hydroDataset[var].dims:
                hydroDataset[var] = hydroDataset[var].isel(longitude=0)

        # Check for NaN values
        all_nan = True
        for var in hydroDataset.data_vars:
            if not np.isnan(hydroDataset[var].values).all():
                all_nan = False
                break
        if all_nan:
            raise ValueError("All the hydrodynamic data is NaN. Select a different point.")
        
        # Write to netCDF
        if writeNetCDF:
            hydroDataset.to_netcdf(os.path.join(folder, "hydroData.nc"))
    
        return hydroDataset
    

    def getDischargeData(self, writeNetCDF=True, overwrite=False, folder="predictors"):
        """
        Loads the discharge data
        
        :param writeNetCDF: bool, whether to write the discharge data to a netCDF file
        :param overwrite: bool, whether to overwrite the discharge data file
        :param folder: str, folder where to store the discharge data
        
        :return: xarray.Dataset, discharge data
        """

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder, "dischargeData.nc")):
            return xr.open_dataset(os.path.join(folder, "dischargeData.nc"))
        
        # Import discharge mat file
        data = sio.loadmat(self.config["predictors"]["discharge"])
        time = datenumToDatetime(data["tsim_Puntal"])
        timeIndex = pd.DatetimeIndex(time)
        discharge = data["Qsim_Puntal"]

        # Interpolate daily discharge data to hourly
        newTime = pd.date_range(start=time[0], end=time[-1], freq="H")
        newDischarge = np.interp(newTime, timeIndex, discharge.flatten())

        # Trim time to the temporal extension
        if isinstance(self.tempExt, list):
            idx = np.where((newTime >= self.tempExt[0][0].to_pydatetime()) & (newTime <= self.tempExt[-1][1].to_pydatetime()))[0]
        else:
            idx = np.where((newTime >= self.tempExt[0].to_pydatetime()) & (newTime <= self.tempExt[1].to_pydatetime()))[0]
        newTime = newTime[idx]
        newDischarge = newDischarge[idx]

        # Create xarray.Dataset
        ds = xr.Dataset(
            {
                "discharge": (("time",), newDischarge)
            },
            coords={"time": pd.to_datetime(newTime)},
        )

        # Write to netCDF
        if writeNetCDF:
            ds.to_netcdf(os.path.join(folder, "dischargeData.nc"))

        return ds


    def getTidalRangeData(self, writeNetCDF=True, overwrite=False, folder="predictors"):
        """
        Loads the tidal range data
        
        :param writeNetCDF: bool, whether to write the tidal range data to a netCDF file
        :param overwrite: bool, whether to overwrite the tidal range data file
        :param folder: str, folder where to store the tidal range data
        
        :return: xarray.Dataset, tidal range data
        """

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder, "tidalRangeData.nc")):
            return xr.open_dataset(os.path.join(folder, "tidalRangeData.nc"))
                                   
        # Import tide gauge mat file
        data = sio.loadmat(self.config["predictors"]["tidalRange"])
        time = np.array(datenumToDatetime(data["Date"]))

        # If temporal extension is within time range, use the data
        if isinstance(self.tempExt, list):
            maxTime = np.max([endExt for _, endExt in self.tempExt])
            minTime = np.min([startExt for startExt, _ in self.tempExt])
        else:
            maxTime = self.tempExt[1]
            minTime = self.tempExt[0]
        if time[-1] >= maxTime and time[0] <= minTime:
            astronomicTide = data["Ma"]
            # Trim time to the temporal extension
            if isinstance(self.tempExt, list):
                idx = np.where((time >= self.tempExt[0][0].to_pydatetime()) & (time <= self.tempExt[-1][1].to_pydatetime()))[0]
            else:
                idx = np.where((time >= self.tempExt[0].to_pydatetime()) & (time <= self.tempExt[1].to_pydatetime()))[0]
            
            time = time[idx]
            astronomicTide = astronomicTide[idx]
        
        else:
            baseCatalogURL = "http://opendap.puertos.es/thredds/catalog/tidegauge_san2/"
            baseURL = "http://opendap.puertos.es/thredds/dodsC/"
            tideGaugeURLs = self.getThreddsURLs(baseCatalogURL, baseURL)
            # Remove all the URLs that contain the word "analysis"
            tideGaugeURLs = [url for url in tideGaugeURLs if "analysis" not in url]
            with xr.open_mfdataset(tideGaugeURLs, decode_times=True) as ds:
                # Select variables of interest
                ds = ds[["SLEV", "SLEV_QC"]]
                # Remove the data where SLEV_QC is not 1 or 2
                ds = ds.where(ds.SLEV_QC.isin([1, 2]))
                # Handle NaN values
                ds = ds.chunk(dict(TIME=-1))
                ds = ds.interpolate_na("TIME", method="linear")
                # Get a moving mean of 5 minutes
                ds = ds.rolling(TIME=600, center=True).mean()
                # Resample to hourly frequency
                ds = ds.resample(TIME="h").mean()

                time = ds.TIME.values
                tide = np.squeeze(ds.SLEV.values)

                coef = solve(time, tide, lat=43.46, trend=False)

                astronomicTide = reconstruct(time, coef).h

        
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
    

    def getThreddsURLs(self, baseCatalogURL, baseURL):
        """
        Gets a list of URLs to download the tide gauge data from Meteogalicia or Puertos del Estado Thredds server.

        :param baseCatalogURL: str, base URL of the catalog
        :param baseURL: str, base URL of the data

        :return: list, URLs to download the wind or tide gauge data
        """

        # Expand the temporal extension one hour
        if isinstance(self.tempExt, list):
            iniDates = list()
            endDates = list()
            for i in range(len(self.tempExt)):
                iniDates.append(self.tempExt[i][0] - pd.Timedelta(hours=1))
                endDates.append(self.tempExt[i][1] + pd.Timedelta(hours=1))
        else:
            iniDates = self.tempExt[0] - pd.Timedelta(hours=1)
            endDates = self.tempExt[1] + pd.Timedelta(hours=1)

        # Get the files
        urls = []
        if isinstance(iniDates, list):
            for iniDate, endDate in zip(iniDates, endDates):
                currentDate = iniDate
                while currentDate.year < endDate.year or (currentDate.year == endDate.year and currentDate.month <= endDate.month):
                    yearMonthCatalogUrl = baseCatalogURL + f"{currentDate.year}/{currentDate.month:02d}/catalog.xml"
                    month_file_urls = self.getFileUrlsFromMonthCatalog(yearMonthCatalogUrl, baseURL)
                    urls.extend(month_file_urls)
                    # Move to the next month
                    _, days_in_month = calendar.monthrange(currentDate.year, currentDate.month)
                    # Move to the first day of the next month
                    currentDate = currentDate.replace(day=1) + timedelta(days=days_in_month)

        else:
            currentDate = iniDates
            while currentDate.year < endDates.year or (currentDate.year == endDates.year and currentDate.month <= endDates.month):
                yearMonthCatalogUrl = baseCatalogURL + f"{currentDate.year}/{currentDate.month:02d}/catalog.xml"
                month_file_urls = self.getFileUrlsFromMonthCatalog(yearMonthCatalogUrl, baseURL)
                urls.extend(month_file_urls)
                # Move to the next month
                _, days_in_month = calendar.monthrange(currentDate.year, currentDate.month)
                # Move to the first day of the next month
                currentDate = currentDate.replace(day=1) + timedelta(days=days_in_month)

        return sorted(urls)
    

    def getFileUrlsFromMonthCatalog(self, catalogUrl, baseURL):
        """
        Gets the file URLs from a month catalog
        
        :param catalogUrl: str, URL of the catalog
        :param baseURL: str, base URL of the data
        
        :return: list, file URLs
        """
        
        fileUrls = []
        namespace = {'ns': 'http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0'}

        response = requests.get(catalogUrl)
        if response.status_code == 200:
            tree = ET.fromstring(response.text)
            for dataset in tree.findall('.//ns:dataset', namespace):
                try:
                    fileUrls.append(baseURL + dataset.attrib.get('urlPath'))
                except:
                    continue
        return fileUrls

