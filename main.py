import json
import pandas as pd
from auxFunc import concatCamel
from predictStation import PredictStation

if __name__ == "__main__":
    # Load model
    configFilePath = "config.json"
    predictStation = PredictStation(configFilePath)

    # Train model
    predictStation.train()

    # Predict new data
    # newHisFile = "C:\\TFM\\input\\recibido_JaviG\\corrientes_bahia\\0_hidros_Mirko\\04_2022\\Santander_his.nc"
    # y, yPred = predictStation.predict(newHisFile=newHisFile, removeTimesteps=168)

    # Predict test data
    y, yPred = predictStation.predict(predMatrix=predictStation.model.predMatrix if not isinstance(predictStation.model, list) else predictStation.model[0].predMatrix)

    if isinstance(predictStation.model, list):
        # Merge y variables
        for i, model in enumerate(predictStation.model):
            if i == 0:
                continue
            else:
                y = pd.concat([y, predictStation.model[i].predMatrix.yTest], axis=1)
    
    # Remove rows with NaN values
    y = y.dropna()


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