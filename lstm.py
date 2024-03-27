from predictors import Predictors
from predictands import Predictands
from predictionMatrix import PredictionMatrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

configFilePath = "config.json"
predictors = Predictors(configFilePath)
predictands = Predictands(configFilePath)
predMatrix = PredictionMatrix(configFilePath, predictors, predictands)

# Convert xTrain and xTest to a pandas DataFrame
xTrain = pd.DataFrame(predMatrix.xTrain, index=predMatrix.yTrain.index)
xTest = pd.DataFrame(predMatrix.xTest, index=predMatrix.yTest.index)

# Reshape xTrain and xTest for the LSTM model
xTrain = xTrain.values.reshape(xTrain.shape[0], 1, xTrain.shape[1])
xTest = xTest.values.reshape(xTest.shape[0], 1, xTest.shape[1])

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(xTrain.shape[1], xTrain.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=3))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Fit the model
history = model.fit(xTrain, predMatrix.yTrain, epochs=50, batch_size=32, validation_data=(xTest, predMatrix.yTest), verbose=1)

# Train the model
earlyStopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(xTrain, predMatrix.yTrain, epochs=100, batch_size=32, validation_data=(xTest, predMatrix.yTest), callbacks=[earlyStopping])

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
yPred = model.predict(x.values.reshape(x.shape[0], 1, x.shape[1]))

# Plot original data and predicted data, removing the first 48 hours
plt.subplot(2, 1, 1)
plt.plot(y.index[48:], y["u_x"][48:], label="y")
plt.plot(y.index[48:], yPred[48:, 0], label="yPred")
plt.title("Original and Predicted u_x")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y.index[48:], y["u_y"][48:], label="y")
plt.plot(y.index[48:], yPred[48:, 1], label="yPred")
plt.title("Original and Predicted u_y")
plt.legend()

# # Predict the test data
# yPredTest = model.predict(xTest)

# # Plot yTest and yPredTest
# plt.subplot(2, 1, 1)
# plt.plot(predMatrix.yTest.index, predMatrix.yTest["u_x"], label="yTest")
# plt.plot(predMatrix.yTest.index, yPredTest[:, 0], label="yPredTest")
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(predMatrix.yTest.index, predMatrix.yTest["u_y"], label="yTest")
# plt.plot(predMatrix.yTest.index, yPredTest[:, 1], label="yPredTest")
# plt.legend()
# plt.show()

print("fin")