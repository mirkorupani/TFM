import json
from predictors import Predictors
from predictands import Predictands
from predictionMatrix import PredictionMatrix
import pandas as pd
from time import time
from auxFunc import concatCamel


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
            if self.config["model"]["lstm"]["differentNetworks"] is None:
                self.model = RnnLstm(self.configFilePath, predMatrix)
            else:
                self.model = []
                for i, var in enumerate(self.config["model"]["lstm"]["differentNetworks"]):
                    yTrainCols = predMatrix.yTrain.columns[predMatrix.yTrain.columns.isin(var)]
                    newPredMatrix = predMatrix.copy()
                    newPredMatrix.yTrain = predMatrix.yTrain[yTrainCols]
                    newPredMatrix.yTest = predMatrix.yTest[yTrainCols]
                    modelName = f"sta{self.config['predictands']['station']}Model{concatCamel(var)}"
                    print(f"Training network {i+1} of {len(self.config['model']['lstm']['differentNetworks'])}: {var[0]}")
                    self.model.append(RnnLstm(self.configFilePath, newPredMatrix, modelName=modelName, nOutput=len(yTrainCols)))
        
        elapsedTime = time() - elapsedTime
        print(f"Training time: {elapsedTime/60} minutes")
    

    def predict(self, predMatrix = None, newHisFile=None, newPredictorsFolder="newPredictors", newPredictandsFolder="newPredictands", removeTimesteps=None):
        
        if predMatrix is not None and newHisFile is None:
            x = predMatrix.xTest
            y = predMatrix.yTest
        elif predMatrix is None and newHisFile is not None:
            newPredictors = Predictors(self.configFilePath, hisFile=newHisFile, folder=newPredictorsFolder)
            newPredictands = Predictands(self.configFilePath, hisFile=newHisFile, folder=newPredictandsFolder)
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
                for i, var in enumerate(self.config["model"]["lstm"]["differentNetworks"]):
                    yPred[var] = self.model[i].model.predict(xSeq)
            
            # Drop the first nInput-1 hours
            y = y.iloc[self.model.nTimesteps-1:]
            
            # Convert yPred to a DataFrame
            if self.config["model"]["lstm"]["differentNetworks"] is None:
                yPred = pd.DataFrame(yPred, index=y.index, columns=y.columns)
        
        return y, yPred
    

    def plotResults(self, y, yPred, startIdx=0, title=None, savePath=None, waterlevel=False, show=True, saveMetrics=True, saveMetricsPath="metrics.json"):
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
    newHisFile = "C:\\TFM\\input\\recibido_JaviG\\corrientes_bahia\\0_hidros_Mirko\\04_2022\\Santander_his.nc"
    y, yPred = predictStation.predict(newHisFile=newHisFile, removeTimesteps=168)

    # # Predict test data
    # y, yPred = predictStation.predict(predMatrix=predictStation.model.predMatrix)

    # Plot results
    predictStation.plotResults(y, yPred, savePath=f"C:\\TFM\\results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}Validation.png", waterlevel=True, saveMetricsPath=f"C:\\TFM\\results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}ValidationMetrics.json")

    # Save config
    predictStation.saveConfig(f"C:\\TFM\\results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}Config.json")

    # If adaBoost, export best parameters to csv
    if predictStation.config["model"]["method"] == "adaboost":
        bestParams = pd.DataFrame()
        for var in predictStation.model.model.keys():
            bestParams[var] = predictStation.model.model["u_x"].best_params_
        bestParams.to_csv(f"C:\\TFM\\results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}BestParams.csv")
    
    # If lstm, export hyperparameters to json
    if predictStation.config["model"]["method"] == "lstm":
        if isinstance(predictStation.model, list):
            for (i, model), var in zip(enumerate(predictStation.model), predictStation.config["model"]["lstm"]["differentNetworks"]):
                bestHP = model.bestHP
                if bestHP is not None:
                    modelName = f"sta{predictStation.config['predictands']['station']}Model{concatCamel(var)}"
                    filePath = f"C:\\TFM\\results\\{predictStation.config['model']['method']}\\sta{modelName}Hyperparameters.json"
                    hp_dict = bestHP.get_config()  # Convert the HyperParameters to a dictionary
                    with open(filePath, "w") as f:
                        json.dump(hp_dict, f, indent=4)
        else:
            bestHP = predictStation.model.bestHP
            if bestHP is not None:
                filePath = f"C:\\TFM\\results\\{predictStation.config['model']['method']}\\sta{predictStation.config['predictands']['station']}Hyperparameters.json"
                hp_dict = bestHP.get_config()  # Convert the HyperParameters to a dictionary
                with open(filePath, "w") as f:
                    json.dump(hp_dict, f, indent=4)