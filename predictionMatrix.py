import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

class PredictionMatrix():
    """Class to get the prediction matrix for the model"""


    def __init__(self, config, predictors, predictands):
        """
        :param config: str, path to the configuration file or dictionary with the configuration
        :param predictors: Predictors, predictors object
        :param predictands: Predictands, predictands object
        
        :return: None
        """
        
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
                self.config = json.load(f)
        self.predictors = predictors
        self.predictands = predictands
        self.x, self.y = self.getPredMatrix()
        self.preprocessData()


    def getPredMatrix(self, newPredictors=False, newPredictands=False, removeTimesteps=None):
        """
        Gets the prediction matrix to be used in the model
        
        :param newPredictors: Predictors, new predictors object
        :param newPredictands: Predictands, new predictands object
        :param removeTimesteps: int, number of time steps to remove
        
        :return: x (pd.DataFrame) prediction matrix if newPredictors and newPredictands are False
        :return: x, y (pd.DataFrame, pd.DataFrame) prediction matrix and predictands in any other case
        """

        # Time steps to remove
        if removeTimesteps is None:
            if not newPredictors:
                removeTimesteps = self.config["predictands"]["removeTimesteps"]

        predictors = self.predictors if not newPredictors else newPredictors
            
        # Get the time series
        if isinstance(predictors.tempExt, list):
            # Concatenate the time series
            timeSeries = pd.date_range(predictors.tempExt[0][0], predictors.tempExt[0][1], freq="h")
            for i in range(1, len(predictors.tempExt)):
                timeSeries = timeSeries.append(pd.date_range(predictors.tempExt[i][0], predictors.tempExt[i][1], freq="h"))
            # Remove duplicates
            timeSeries = timeSeries[~timeSeries.duplicated()]

        else:
            timeSeries = pd.date_range(predictors.tempExt[0], predictors.tempExt[1], freq="h")

        # Predictors
        x = pd.DataFrame(index=timeSeries)

        # Add the wind data
        if self.config["predictors"]["wind"] == "meteogalicia":
            # If wind data is shorter than x, trim wind data
            if len(predictors.windData.time) != len(x):
                predictors.windData = predictors.windData.isel(time=slice(0, len(x)))
            x["u10"] = predictors.windData.u
            x["v10"] = predictors.windData.v
            x["slp"] = predictors.windData.mslp
        elif self.config["predictors"]["wind"] == None:
            pass
        else:
            raise ValueError("Wind data source not recognized")

        # Add the current data
        # If hydro data is longer than x, trim hydro data
        if len(predictors.hydroData.time) != len(x):
            predictors.hydroData = predictors.hydroData.isel(time=slice(0, len(x)))
        for varList in self.config["predictors"]["hydro"]["variables"]:
            for var in varList:
                x[var] = predictors.hydroData[var].values

        # Add the discharge data
        if self.config["predictors"]["discharge"] is not None:
            # If discharge data is longer than x, trim discharge data
            if len(predictors.dischargeData.time) != len(x):
                predictors.dischargeData = predictors.dischargeData.isel(time=slice(0, len(x)))
            x["discharge"] = predictors.dischargeData.discharge

        # Add the tidal range data
        if self.config["predictors"]["tidalRange"] is not None:
            # If tidal range data is shorter than x, trim x
            if len(predictors.tidalRangeData.time) < len(x):
                x = x.iloc[:len(predictors.tidalRangeData.tidalRange)]
            # If tidal range data is longer than x, trim tidal range data
            elif len(predictors.tidalRangeData.time) > len(x):
                predictors.tidalRangeData = predictors.tidalRangeData.isel(time=slice(0, len(x)))
            x["tidalRange"] = predictors.tidalRangeData.tidalRange

        if newPredictors and not newPredictands:
            if removeTimesteps is not None:
                return x.iloc[removeTimesteps-1:]
            return x
        
        else:
            predictands = self.predictands if not newPredictands else newPredictands
            # Predictands
            y = pd.DataFrame(index=timeSeries)
            # If predictands are longer than y, trim predictands
            if len(predictands.predictands.time) != len(y):
                predictands.predictands = predictands.predictands.isel(time=slice(0, len(y)))
            for var in self.config["predictands"]["variables"]:
                y[var] = predictands.predictands[var]
            
            # If predictors are shorter than predictands, trim predictands
            if len(x) != len(y):
                y = y.iloc[:len(x)]

            if removeTimesteps is not None:
                return x.iloc[removeTimesteps-1:], y.iloc[removeTimesteps-1:]
            
            return x, y
    

    def preprocessData(self, newPredMatrix=False):
        """
        Preprocesses the data
        
        :param newPredMatrix: bool, whether to preprocess a new prediction matrix
        
        :return: None
        """

        if newPredMatrix is False:

            # Split the data
            self.xTrain, self.xTest, self.yTrain, self.yTest = self.splitData()

            # Scale the data
            self.scaler, self.xTrain, self.xTest = self.scaleData()

            # Dimensionality reduction
            if self.config["preprocess"]["dimReduction"]["method"] is not None:
                self.model, self.xTrain, self.xTest = self.dimReduction()
        
        else:
            if self.config["preprocess"]["dimReduction"]["method"] is not None:
                return self.model.transform(self.scaler.transform(newPredMatrix))
            else:
                return self.scaler.transform(newPredMatrix)
    

    def splitData(self):
        """
        Splits the data into training and testing sets
        
        :return: tuple, (xTrain, xTest, yTrain, yTest)
        """

        # Split the data
        if self.config["preprocess"]["trainTestSplit"]["method"] == "temporal":
            split = int(len(self.x) * self.config["preprocess"]["trainTestSplit"]["testSize"])
            xTrain, xTest = self.x.iloc[:-split], self.x.iloc[-split:]
            yTrain, yTest = self.y.iloc[:-split], self.y.iloc[-split:]
        
        elif self.config["preprocess"]["trainTestSplit"]["method"] == None:
            return self.x, None, self.y, None
        
        return xTrain, xTest, yTrain, yTest
    

    def scaleData(self):
        """
        Scales the data
        
        :return: tuple, (scaler, xTrain, xTest)
        """

        # Scale the data
        if self.config["preprocess"]["scale"]["method"] == "standard":
            scaler = StandardScaler()
            xTrain = scaler.fit_transform(self.xTrain)
            if self.xTest is not None:
                xTest = scaler.transform(self.xTest)
        
        elif self.config["preprocess"]["scaling"]["method"] == None:
            return self.xTrain, self.xTest
        
        if self.xTest is not None:
            return scaler, xTrain, xTest
        else:
            return scaler, xTrain, None
    

    def dimReduction(self):
        """
        Performs dimensionality reduction on the data
        
        :return: tuple, (model, xTrain, xTest)
        """

        # Dimensionality reduction
        if self.config["preprocess"]["dimReduction"]["method"] == "pca":
            model = PCA(n_components=self.config["preprocess"]["dimReduction"]["nComponents"])
            xTrain = model.fit_transform(self.xTrain)
            if self.xTest is not None:
                xTest = model.transform(self.xTest)
        
        elif self.config["preprocess"]["dimReduction"]["method"] == "isomap":
            model = Isomap(n_components=self.config["preprocess"]["dimReduction"]["nComponents"])
            xTrain = model.fit_transform(self.xTrain)
            if self.xTest is not None:
                xTest = model.transform(self.xTest)

        elif self.config["preprocess"]["dimReduction"]["method"] == None:
            return self.yTrain, self.yTest
        
        if self.xTest is not None:
            return model, xTrain, xTest
        else:
            return model, xTrain, None
    

    def copy(self):
        """
        Copies the prediction matrix
        
        :return: PredictionMatrix, copy of the prediction matrix
        """

        return PredictionMatrix(self.config, self.predictors, self.predictands)