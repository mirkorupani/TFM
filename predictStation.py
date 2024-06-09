import json
from predictors import Predictors
from predictands import Predictands
from predictionMatrix import PredictionMatrix
import pandas as pd
from time import time
from auxFunc import concatCamel


class PredictStation():
    """Class to predict the values of the predictands"""
    
    def __init__(self, config):
        """
        Initializes the class
        :param configFilePath: str, path to the configuration file
        
        :return: None
        """

        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
                self.config = json.load(f)
        self.model = None
        

    def train(self, hyperparameters=None):
        """
        Trains the model

        :return: None
        """
        predictors = Predictors(self.config)
        predictands = Predictands(self.config)
        predMatrix = PredictionMatrix(self.config, predictors, predictands)

        print("Training model...")
        elapsedTime = time()

        if self.config["model"]["method"] == "analogues":
            from analogues import Analogues
            self.model = Analogues(self.config, predMatrix)

        elif self.config["model"]["method"] == "adaboost":
            from adaboost import AdaBoost
            self.model = AdaBoost(self.config, predMatrix)

        elif self.config["model"]["method"] == "lstm":
            from lstm import RnnLstm
            if self.config["model"]["lstm"]["differentNetworks"] is None:
                self.model = RnnLstm(self.config, predMatrix, hyperparameters=hyperparameters)
            else:
                self.model = []
                for i, var in enumerate(self.config["model"]["lstm"]["differentNetworks"]):
                    yTrainCols = predMatrix.yTrain.columns[predMatrix.yTrain.columns.isin(var)]
                    newPredMatrix = predMatrix.copy()
                    newPredMatrix.yTrain = predMatrix.yTrain[yTrainCols]
                    newPredMatrix.yTest = predMatrix.yTest[yTrainCols]
                    modelName = f"sta{self.config['predictands']['station']}Model{concatCamel(var)}"
                    print(f"Training network {i+1} of {len(self.config['model']['lstm']['differentNetworks'])}: {var[0]}")
                    self.model.append(RnnLstm(self.config, newPredMatrix, modelName=modelName, nOutput=len(yTrainCols), hyperparameters=hyperparameters))
        
        elapsedTime = time() - elapsedTime
        print(f"Training time: {elapsedTime/60} minutes")
    

    def predict(self, predMatrix = None, newHisFile=None, newPredictorsFolder="newPredictors", newPredictandsFolder="newPredictands", removeTimesteps=None):
        """
        Predicts the values of the predictands
        
        :param predMatrix: PredictionMatrix, prediction matrix
        :param newHisFile: str, path to the new history file
        :param newPredictorsFolder: str, folder where the new predictors are stored
        :param newPredictandsFolder: str, folder where the new predictands are stored
        :param removeTimesteps: int, number of timesteps to remove
        
        :return: pandas.DataFrame, original values
        :return: pandas.DataFrame, predicted values
        """
        
        if predMatrix is not None and newHisFile is None:
            x = predMatrix.xTest
            y = predMatrix.yTest
        elif predMatrix is None and newHisFile is not None:
            newPredictors = Predictors(self.config, hisFile=newHisFile, folder=newPredictorsFolder)
            newPredictands = Predictands(self.config, hisFile=newHisFile, folder=newPredictandsFolder)
            if self.config["model"]["lstm"]["differentNetworks"] is None:
                x, y = self.model.predMatrix.getPredMatrix(newPredictors=newPredictors, newPredictands=newPredictands, removeTimesteps=removeTimesteps)
            else:
                x, y = self.model[0].predMatrix.getPredMatrix(newPredictors=newPredictors, newPredictands=newPredictands, removeTimesteps=removeTimesteps)
            x = self.model[0].predMatrix.preprocessData(x)
        else:
            raise ValueError("Either predMatrix or newHisFile must be provided")
        
        # Convert x to a pandas DataFrame
        x = pd.DataFrame(x, index=y.index)
        
        if self.config["model"]["method"] == "analogues":
            yPred = self.model.regressor.predict(x)
            # Convert yPred to a DataFrame
            yPred = pd.DataFrame(yPred, index=y.index, columns=y.columns)
        
        elif self.config["model"]["method"] == "adaboost":
            yPred = pd.DataFrame(index=y.index)
            for var in y.columns:
                yPred[var] = self.model.model[var].predict(x)
        
        elif self.config["model"]["method"] == "lstm":
            # Create sequence
            xSeq = self.model.preprocessData(x) if self.config["model"]["lstm"]["differentNetworks"] is None else self.model[0].preprocessData(x)

            # Predict
            if self.config["model"]["lstm"]["differentNetworks"] is None:
                yPred = self.model.model.predict(xSeq)
            else:
                yPred = pd.DataFrame(index=y.index)
                # Remove the first nInput-1 hours
                yPred = yPred.iloc[self.model[0].nTimesteps-1:]
                for i, var in enumerate(self.config["model"]["lstm"]["differentNetworks"]):
                    yPred[var] = self.model[i].model.predict(xSeq)
            
            # Drop the first nInput-1 hours
            y = y.iloc[self.model.nTimesteps-1:] if self.config["model"]["lstm"]["differentNetworks"] is None else y.iloc[self.model[0].nTimesteps-1:]
            
            # Convert yPred to a DataFrame
            if self.config["model"]["lstm"]["differentNetworks"] is None:
                yPred = pd.DataFrame(yPred, index=y.index, columns=y.columns)
        
        return y, yPred
    

    def plotResults(self, y, yPred, startIdx=0, title=None, savePath=None, waterlevel=False, show=True, saveMetrics=True, saveMetricsPath="metrics.json"):
        """
        Plots the results

        :param y: pandas.DataFrame, original values
        :param yPred: pandas.DataFrame, predicted values
        :param startIdx: int, number of timesteps to remove from the plot and the validation metrics
        :param title: str, title of the plot
        :param savePath: str, path to save the plot
        :param waterlevel: bool, if True, plot the waterlevel
        :param show: bool, if True, show the plot
        :param saveMetrics: bool, if True, save the metrics to a json file
        :param saveMetricsPath: str, path to save the metrics

        :return: None
        """
        from plotFunct import combinedPlots
        if saveMetrics:
            metrics = combinedPlots(y, yPred, startIdx=startIdx, title=title, savePath=savePath, waterlevel=waterlevel, returnMetrics=True)
            # Convert float32 to serializable
            for key in metrics.keys():
                metrics[key] = {var: metrics[key][var].item() for var in metrics[key].keys()}
            with open(saveMetricsPath, "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            combinedPlots(y, yPred, startIdx=startIdx, title=title, savePath=savePath, waterlevel=waterlevel)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    

    def saveConfig(self, savePath="config.json"):
        """
        Saves the configuration file
        
        :param savePath: str, path to save the configuration file
        
        :return: None
        """
        with open(savePath, "w") as f:
            json.dump(self.config, f, indent=4)
        
        if self.config["model"]["method"] == "adaboost":
            bestParams = pd.DataFrame()
            for var in self.model.model.keys():
                bestParamsDict = self.model.model[var].best_params_
                bestParamsDict = {key: [value] for key, value in bestParamsDict.items()}
                bestParams[var] = pd.DataFrame(bestParamsDict).T