import json
from predictors import Predictors
from predictands import Predictands
from predictionMatrix import PredictionMatrix
import pandas as pd
from time import time

class PredictSation():
    
    def __init__(self, configFilePath):
        self.configFilePath = configFilePath
        with open(configFilePath) as f:
            self.config = json.load(f)
        self.model = None
        

    def train(self):
        predictors = Predictors(self.configFilePath)
        predictands = Predictands(self.configFilePath)
        predMatrix = PredictionMatrix(self.configFilePath, predictors, predictands)

        if self.config["model"]["method"] == "analogues":
            from analogues import Analogues
            self.model = Analogues(self.configFilePath, predMatrix)

        elif self.config["model"]["method"] == "adaboost":
            from adaboost import AdaBoost
            self.model = AdaBoost(self.configFilePath, predMatrix)

        elif self.config["model"]["method"] == "lstm":
            from lstm import RnnLstm
            self.model = RnnLstm(self.configFilePath, predMatrix)
    

    def predict(self, newHisFile, newPredictorsFolder="newPredictors", newPredictandsFolder="newPredictands"):
        newPredictors = Predictors(self.configFilePath, hisFile=newHisFile, folder=newPredictorsFolder)
        newPredictands = Predictands(self.configFilePath, hisFile=newHisFile, folder=newPredictandsFolder)
        x, y = self.model.predMatrix.getPredMatrix(newPredictors=newPredictors, newPredictands=newPredictands)
        x = self.model.predMatrix.preprocessData(x)
        # Convert x to a pandas DataFrame
        x = pd.DataFrame(x, index=y.index)
        
        if self.config["model"]["method"] == "analogues":
            yPred = self.model.predict(x)
        
        elif self.config["model"]["method"] == "adaboost":
            yPred = pd.DataFrame(index=y.index)
            for var in y.columns:
                yPred[var] = self.model.model[var].predict(x)
        
        elif self.config["model"]["method"] == "lstm":
            # Create sequence
            xSeq = self.model.preprocessData(x)
            # Predict
            yPred = self.model.model.predict(xSeq)
            # Drop the first nInput-1 hours
            y = y.iloc[self.model.nTimesteps-1:]
            # Convert yPred to a DataFrame
            yPred = pd.DataFrame(yPred, index=y.index, columns=y.columns)
        
        return y, yPred
    

    def plotResults(self, y, yPred, startIdx=0, title=None, savePath=None, waterlevel=False, show=True):
        from plotFunct import combinedPlot
        combinedPlot(y, yPred, startIdx=startIdx, title=title, savePath=savePath, waterlevel=waterlevel)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    

    def saveConfig(self, savePath="config.json"):
        with open(savePath, "w") as f:
            json.dump(self.model.config, f, indent=4)
        
        if self.config["model"]["method"] == "adaboost":
            bestParams = pd.DataFrame()
            for var in self.model.model[var].columns:
                bestParamsDict = self.model.model[var].best_params_
                bestParamsDict = {key: [value] for key, value in bestParamsDict.items()}
                bestParams[var] = pd.DataFrame(bestParamsDict).T
            bestParams.to_csv(f"{savePath[:-5]}BestParams.csv")


if __name__ == "__main__":
    # Load model
    configFilePath = "config.json"
    predictStation = PredictSation(configFilePath)

    # Train model
    print("Training model...")
    elapsedTime = time()
    predictStation.train()
    elapsedTime = time() - elapsedTime
    print(f"Training time: {elapsedTime/60} minutes")

    # Predict new data
    newHisFile = "F:\\recibido_JaviG\\corrientes_bahia\\validacion_2012\\Santander_his.nc"
    y, yPred = predictStation.predict(newHisFile)

    # Plot results
    predictStation.plotResults(y, yPred, startIdx=48, savePath=f"F:\\TFM_results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}Validation.png")

    # Save config
    predictStation.saveConfig(f"F:\\TFM_results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}Config.json")