import json
import xarray as xr
import os

class Predictands():
    """Class to get and load the predictands for the model"""


    def __init__(self, config, hisFile=None, folder="predictands"):
        
        with open(config) as f:
            self.config = json.load(f)
        
        if hisFile is None:
            folder = self.config["predictands"]["predictandsFolder"]
            hisFile = self.config["predictands"]["hisFile"]
        
        self.predictands = self.getPredictands(hisFile, folder=folder)
    
    def getPredictands(self, hisFile, writeNetCDF=True, overwrite=False, folder="predictands"):
        """Gets the predictands
        :return: pandas.DataFrame, predictands"""

        # Check if file exists
        if not overwrite and os.path.exists(os.path.join(folder,"predictands.nc")):
            return xr.open_dataset(os.path.join(folder,"predictands.nc"))
        
        with xr.open_dataset(hisFile) as ds:
            predictands = ds[self.config["predictands"]["variables"]].isel(Station=self.config["predictands"]["station"]).sel(Layer=self.config["predictands"]["sigmaLayer"])

            # Get hourly data
            if self.config["predictands"]["resample"] == "mean":
                predictands = predictands.resample(time="H").mean()
            else:
                raise ValueError("Resample method not recognized")
        
        if writeNetCDF:
            predictands.to_netcdf(os.path.join(folder,"predictands.nc"))
        
        return predictands