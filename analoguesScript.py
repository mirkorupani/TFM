from predictors import Predictors
from predictands import Predictands
from predictionMatrix import PredictionMatrix
from analogues import Analogues
import pandas as pd
import matplotlib.pyplot as plt
import json
from plotFunct import combinedPlot


configFilePath = "config.json"

predictors = Predictors(configFilePath)
predictands = Predictands(configFilePath)
predMatrix = PredictionMatrix(configFilePath, predictors, predictands)
analogues = Analogues(configFilePath, predMatrix)

# Predict new data
newHisFile = "F:\\recibido_JaviG\\corrientes_bahia\\validacion_2012\\Santander_his.nc"
newPredictorsFolder = "F:\\TFM\\newPredictors"
newPredictandsFolder = "F:\\TFM\\newPredictands"
newPredictors = Predictors(configFilePath, hisFile=newHisFile, folder=newPredictorsFolder)
newPredictands = Predictands(configFilePath, hisFile=newHisFile, folder=newPredictandsFolder)
x, y = predMatrix.getPredMatrix(newPredictors=newPredictors, newPredictands=newPredictands)
x = predMatrix.preprocessData(x)
# Convert x to a pandas DataFrame
x = pd.DataFrame(x, index=y.index)
yPred = analogues.predict(x)

# Plot original data and predicted data, removing the first 48 hours
with open(configFilePath) as f:
    config = json.load(f)
    nAnalogues = config["model"]["analogues"]["nAnalogues"]
    clustering = config["model"]["analogues"]["clustering"]
    neighbors = config["model"]["analogues"]["nNeighbors"]
combinedPlot(y, yPred, startIdx=48, title=f"{nAnalogues} analogues - {clustering} clustering - {neighbors} neighbors")
plt.show()

print("fin")