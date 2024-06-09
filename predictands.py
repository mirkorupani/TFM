import json
import xarray as xr
import os

class Predictands():
    """Class to get and load the predictands for the model"""


    def __init__(self, config, hisFile=None, folder="predictands"):
        """
        :param config: str, path to the configuration file or dictionary with the configuration
        :param hisFile: str, path to the history file
        :param folder: str, folder where the predictands are stored
        
        :return: None
        """
        
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
                self.config = json.load(f)
        
        if hisFile is None:
            folder = self.config["predictands"]["predictandsFolder"]
            hisFile = self.config["predictands"]["hisFile"]
        
        self.predictands = self.getPredictands(hisFile, folder=folder)
    
    def getPredictands(self, hisFile, writeNetCDF=True, overwrite=False, folder="predictands"):
        """
        Loads the predictands from the history file
        
        :param hisFile: str, path to the history file
        :param writeNetCDF: bool, whether to write the predictands to a NetCDF file
        :param overwrite: bool, whether to overwrite the NetCDF file if it already exists
        :param folder: str, folder where to store the predictands
        
        :return: xarray.Dataset, predictands dataset
        """

        station = self.config["predictands"]["station"]
        filePath = os.path.join(folder,f"Sta{station}predictands.nc")

        # Check if file exists
        if not overwrite and os.path.exists(filePath):
            return xr.open_dataset(filePath)
        
        if isinstance(hisFile, list):

            # Load datasets and preprocess
            datasets = [xr.open_dataset(file) for file in hisFile]

            # Handle overlapping datasets
            combined_ds = datasets[0]
            for ds in datasets[1:]:
                # Combine datasets, giving precedence to the first dataset
                combined_ds = combined_ds.combine_first(ds)

            # Chunk the combined dataset
            combined_ds = combined_ds.chunk({'time': 5000, 'Station': 1, 'Layer': 1})

            # Select the desired variables and coordinates
            predictands = combined_ds[self.config["predictands"]["variables"]].isel(Station=self.config["predictands"]["station"]).sel(Layer=self.config["predictands"]["sigmaLayer"])


            # with xr.open_mfdataset(hisFile, decode_times=True, chunks="auto", combine="nested", combine_attrs="override", compat="override") as ds:
            #     predictands = ds[self.config["predictands"]["variables"]].isel(Station=self.config["predictands"]["station"]).sel(Layer=self.config["predictands"]["sigmaLayer"])
        else:
            with xr.open_dataset(hisFile) as ds:
                predictands = ds[self.config["predictands"]["variables"]].isel(Station=self.config["predictands"]["station"]).sel(Layer=self.config["predictands"]["sigmaLayer"])
        
        # Round the time values to the nearest second
        predictands["time"] = predictands["time"].dt.round("s")

        # Get hourly data
        if predictands.time.to_index().inferred_freq != "h":
            if self.config["predictands"]["resample"] == "mean":
                predictands = predictands.resample(time="h").mean()
            else:
                raise ValueError("Resample method not recognized")
        
        # Remove NaNs
        # predictands = predictands.dropna("time")
        
        if writeNetCDF:
            predictands.to_netcdf(filePath)
        
        return predictands