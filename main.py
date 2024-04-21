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

        print("Training model...")
        elapsedTime = time()

        if self.config["model"]["method"] == "analogues":
            from analogues import Analogues
            self.model = Analogues(self.configFilePath, predMatrix)

        elif self.config["model"]["method"] == "adaboost":
            from adaboost import AdaBoost
            self.model = AdaBoost(self.configFilePath, predMatrix)

        elif self.config["model"]["method"] == "lstm":
            from lstm import RnnLstm
            self.model = RnnLstm(self.configFilePath, predMatrix)
        
        elapsedTime = time() - elapsedTime
        print(f"Training time: {elapsedTime/60} minutes")
    

    def predict(self, newHisFile, newPredictorsFolder="newPredictors", newPredictandsFolder="newPredictands", removeTimesteps=None):
        newPredictors = Predictors(self.configFilePath, hisFile=newHisFile, folder=newPredictorsFolder)
        newPredictands = Predictands(self.configFilePath, hisFile=newHisFile, folder=newPredictandsFolder)
        x, y = self.model.predMatrix.getPredMatrix(newPredictors=newPredictors, newPredictands=newPredictands, removeTimesteps=removeTimesteps)
        x = self.model.predMatrix.preprocessData(x)
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
            xSeq = self.model.preprocessData(x)
            # Predict
            yPred = self.model.model.predict(xSeq)
            # Drop the first nInput-1 hours
            y = y.iloc[self.model.nTimesteps-1:]
            # Convert yPred to a DataFrame
            yPred = pd.DataFrame(yPred, index=y.index, columns=y.columns)
        
        return y, yPred
    

    def plotResults(self, y, yPred, startIdx=0, title=None, savePath=None, waterlevel=False, show=True, saveMetrics=True, saveMetricsPath="metrics.json"):
        from plotFunct import combinedPlot
        if saveMetrics:
            mae, bias, skillIndex, kolmogorovSmirnov, pearson = combinedPlot(y, yPred, startIdx=startIdx, title=title, savePath=savePath, waterlevel=waterlevel, returnMetrics=True)
            metrics = {"mae": mae, "bias": bias, "skillIndex": skillIndex, "kolmogorovSmirnov": kolmogorovSmirnov, "pearson": pearson}
            # Convert float32 to serializable
            for key in metrics.keys():
                metrics[key] = {var: metrics[key][var].item() for var in metrics[key].keys()}
            with open(saveMetricsPath, "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            combinedPlot(y, yPred, startIdx=startIdx, title=title, savePath=savePath, waterlevel=waterlevel)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    

    def saveConfig(self, savePath="config.json"):
        with open(savePath, "w") as f:
            json.dump(self.model.config, f, indent=4)
        
        if self.config["model"]["method"] == "adaboost":
            bestParams = pd.DataFrame()
            for var in self.model.model.keys():
                bestParamsDict = self.model.model[var].best_params_
                bestParamsDict = {key: [value] for key, value in bestParamsDict.items()}
                bestParams[var] = pd.DataFrame(bestParamsDict).T


if __name__ == "__main__":
    # Load model
    configFilePath = "config.json"
    predictStation = PredictSation(configFilePath)

    # Train model
    predictStation.train()

    # Predict new data
    newHisFile = r"D:\F\recibido_JaviG\corrientes_bahia\validacion_2012\Santander_his.nc"
    y, yPred = predictStation.predict(newHisFile, removeTimesteps=24)

    # Plot results
    predictStation.plotResults(y, yPred, savePath=f"D:\\F\\TFM_results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}Validation.png", waterlevel=True, saveMetricsPath=f"D:\\F\\TFM_results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}ValidationMetrics.json")

    # Save config
    predictStation.saveConfig(f"D:\\F\\TFM_results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}Config.json")

    # If adaBoost, export best parameters to csv
    if predictStation.config["model"]["method"] == "adaboost":
        bestParams = pd.DataFrame()
        for var in predictStation.model.model.keys():
            bestParams[var] = predictStation.model.model["u_x"].best_params_
        bestParams.to_csv(f"D:\\F\\TFM_results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}BestParams.csv")
    
    # If lstm, export hyperparameters to csv
    if predictStation.config["model"]["method"] == "lstm":
        pass