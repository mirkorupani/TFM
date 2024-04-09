import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

class PredictionMatrix():
    """Class to get the prediction matrix for the model"""


    def __init__(self, config, predictors, predictands):
        
        with open(config) as f:
            self.config = json.load(f)
        self.predictors = predictors
        self.predictands = predictands
        self.x, self.y = self.getPredMatrix()
        self.preprocessData()


    def getPredMatrix(self, newPredictors=False, newPredictands=False):
        """Gets the prediction matrix
        :return: pandas.DataFrame, prediction matrix"""

        predictors = self.predictors if not newPredictors else newPredictors
            
        # Get the time series
        if isinstance(predictors.tempExt, list):
            # Concatenate the time series
            timeSeries = pd.date_range(predictors.tempExt[0][0], predictors.tempExt[0][1], freq="h")
            for i in range(1, len(predictors.tempExt)):
                timeSeries = timeSeries.append(pd.date_range(predictors.tempExt[i][0], predictors.tempExt[i][1], freq="h"))
        else:
            timeSeries = pd.date_range(predictors.tempExt[0], predictors.tempExt[1], freq="h")

        # Predictors
        x = pd.DataFrame(index=timeSeries)

        # Add the wind data
        if self.config["predictors"]["wind"] == "meteogalicia":
            x["u10"] = predictors.windData.u
            x["v10"] = predictors.windData.v
            x["slp"] = predictors.windData.mslp
        else:
            raise ValueError("Wind data source not recognized")

        # Add the current data
        for var in self.config["predictors"]["hydro"]["variables"]:
            x[var] = predictors.hydroData[var]

        # Add the discharge data
        pass

        if newPredictors and not newPredictands:
            return x
        
        else:
            predictands = self.predictands if not newPredictands else newPredictands
            # Predictands
            y = pd.DataFrame(index=timeSeries)
            for var in self.config["predictands"]["variables"]:
                y[var] = predictands.predictands[var]

            return x, y
    

    def preprocessData(self, newPredMatrix=False):
        """Preprocesses the data
        :return: tuple, (xTrain, xTest, yTrain, yTest)"""
        
        if newPredMatrix is False:
            # Split the data
            self.xTrain, self.xTest, self.yTrain, self.yTest = self.splitData()

            # Scale the data
            self.scaler, self.xTrain, self.xTest = self.scaleData()

            # Dimensionality reduction
            self.model, self.xTrain, self.xTest = self.dimReduction()
        
        else:
            return self.model.transform(self.scaler.transform(newPredMatrix))
    

    def splitData(self):
        """Splits the data into training and testing sets
        :return: tuple, (xTrain, xTest, yTrain, yTest)"""

        # Split the data
        if self.config["preprocess"]["trainTestSplit"]["method"] == "temporal":
            split = int(len(self.x) * self.config["preprocess"]["trainTestSplit"]["testSize"])
            xTrain, xTest = self.x.iloc[:-split], self.x.iloc[-split:]
            yTrain, yTest = self.y.iloc[:-split], self.y.iloc[-split:]
        
        elif self.config["preprocess"]["trainTestSplit"]["method"] == None:
            return self.x, None, self.y, None
        
        return xTrain, xTest, yTrain, yTest
    

    def scaleData(self):
        """Scales the data
        :return: tuple, (xTrain, xTest)"""

        # Scale the data
        if self.config["preprocess"]["scale"]["method"] == "standard":
            scaler = StandardScaler()
            xTrain = scaler.fit_transform(self.xTrain)
            xTest = scaler.transform(self.xTest)
        
        elif self.config["preprocess"]["scaling"]["method"] == None:
            return self.xTrain, self.xTest
        
        return scaler, xTrain, xTest
    

    def dimReduction(self):
        """Applies dimensionality reduction to the data
        :return: tuple, (xTrain, xTest)"""

        # Dimensionality reduction
        if self.config["preprocess"]["dimReduction"]["method"] == "pca":
            model = PCA(n_components=self.config["preprocess"]["dimReduction"]["nComponents"])
            xTrain = model.fit_transform(self.xTrain)
            xTest = model.transform(self.xTest)
        
        elif self.config["preprocess"]["dimReduction"]["method"] == "isomap":
            model = Isomap(n_components=self.config["preprocess"]["dimReduction"]["nComponents"])
            xTrain = model.fit_transform(self.xTrain)
            xTest = model.transform(self.xTest)

        elif self.config["preprocess"]["dimReduction"]["method"] == None:
            return self.yTrain, self.yTest
        
        return model, xTrain, xTest