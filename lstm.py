# from predictors import Predictors
# from predictands import Predictands
# from predictionMatrix import PredictionMatrix
# import matplotlib.pyplot as plt
# from plotFunct import combinedPlot
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
import keras_tuner as kt
import pandas as pd
import numpy as np
import json


class RnnLstm():
    """Class to train an LSTM model from the prediction matrix"""

    def __init__(self, config, predMatrix, nInput=10, stepSize=1):
        """Constructor of the class"""
        with open(config) as f:
            self.config = json.load(f)
        self.predMatrix = predMatrix
        self.nInput = nInput
        self.stepSize = stepSize
        self.nOutput = len(self.config["predictands"]["variables"])
        self.xTrainSeq, self.xTestSeq = self.preprocessData()
        self.model = self.trainModel()
    

    def preprocessData(self, newData=False):
        if newData is not False:
            xTrain = pd.DataFrame(newData, index=newData.index)
        else:
            # Convert xTrain and xTest to a pandas DataFrame
            xTrain = pd.DataFrame(self.predMatrix.xTrain, index=self.predMatrix.yTrain.index)
            xTest = pd.DataFrame(self.predMatrix.xTest, index=self.predMatrix.yTest.index)

        # Create an empty array to store the input sequences
        xTrainSeq = np.empty((len(range(0, len(xTrain) - self.nInput + 1, self.stepSize)), self.nInput, xTrain.shape[1]))

        # Create the input sequences
        for i, idx in enumerate(range(0, len(xTrain) - self.nInput + 1, self.stepSize)):
            xTrainSeq[i] = xTrain.iloc[idx:idx+self.nInput].values

        if newData is False:
            # Create an empty array to store the input sequences
            xTestSeq = np.empty((len(range(0, len(xTest) - self.nInput + 1, self.stepSize)), self.nInput, xTest.shape[1]))

            # Create the input sequences
            for i, idx in enumerate(range(0, len(xTest) - self.nInput + 1, self.stepSize)):
                xTestSeq[i] = xTest.iloc[idx:idx+self.nInput].values
        
        if newData is not False:
            return xTrainSeq
        else:
            return xTrainSeq, xTestSeq
    

    def trainModel(self):
        configHyperband = self.config["model"]["lstm"]["hyperband"]
        inputShape = (self.nInput, self.predMatrix.xTrain.shape[1])
        model = HyperRegressor(inputShape, self.nOutput, self.config)

        # Define objective
        objective = kt.Objective(configHyperband["objective"], direction=configHyperband["direction"])

        # Define tuner
        tuner = kt.tuners.Hyperband(
            hypermodel=model,
            objective=objective,
            max_epochs=configHyperband["maxEpochs"],
            factor=configHyperband["factor"],
            overwrite = configHyperband["overwrite"],
            directory=configHyperband["directory"],
            project_name=configHyperband["projectName"],
            seed=self.config["randomState"]
        )

        # Search for the best hyperparameters
        tuner.search(self.xTrainSeq, self.predMatrix.yTrain[self.nInput-1::self.stepSize], validation_data=(self.xTestSeq, self.predMatrix.yTest[self.nInput-1::self.stepSize]))

        # Get the best model
        bestHP = tuner.get_best_hyperparameters()[0]
        bestModel = model.build(bestHP)
        
        # Fit the best model
        configEarlyStopping = self.config["model"]["lstm"]["train"]["earlyStopping"]
        bestModel.fit(
            self.xTrainSeq,
            self.predMatrix.yTrain[self.nInput-1::self.stepSize],
            validation_data=(self.xTestSeq, self.predMatrix.yTest[self.nInput-1::self.stepSize]),
            epochs=configHyperband["maxEpochs"],
            callbacks=[EarlyStopping(
                monitor=configEarlyStopping["monitor"],
                patience=configEarlyStopping["patience"],
                restore_best_weights=True
                )]
        )

        return bestModel


# Create hyperregressor class
class HyperRegressor(kt.HyperModel):
        
    def __init__(self, inputShape, nOutput, config):
        self.inputShape = inputShape
        self.nOutput = nOutput
        self.config = config
    

    def build(self, hp):
        model = Sequential()

        configLstm = self.config["model"]["lstm"]["lstmLayers"]
        configDropout = self.config["model"]["lstm"]["dropout"]
        num_lstm_layers = hp.Int('numLstmLayers',
                                 configLstm["minLstmLayers"],
                                 configLstm["maxLstmLayers"])
        for i in range(num_lstm_layers):
            return_sequences = i < num_lstm_layers - 1  # Set return_sequences to True for all except last LSTM layer
            lstm_units = hp.Int('lstmUnits_' + str(i),
                                configLstm["minLstmUnits"],
                                configLstm["maxLstmUnits"],
                                configLstm["stepLstmUnits"])
            dropout_lstm = hp.Boolean('dropoutLstm_' + str(i))
            dropout_rate_lstm = hp.Float('dropoutRateLstm_' + str(i),
                                         min_value=configDropout["minDropout"],
                                         max_value=configDropout["maxDropout"],
                                         step=configDropout["stepDropout"])
            if i == 0:
                model.add(LSTM(units=lstm_units,
                            activation='relu',
                            input_shape=self.inputShape,
                            return_sequences=return_sequences))
            else:
                model.add(LSTM(units=lstm_units,
                            activation='relu',
                            return_sequences=return_sequences))
            model.add(BatchNormalization())
            if dropout_lstm:
                model.add(Dropout(rate=dropout_rate_lstm))

        configDense = self.config["model"]["lstm"]["denseLayers"]
        for i in range(hp.Int('numDenseLayers',
                              configDense["minDenseLayers"],
                              configDense["maxDenseLayers"])):
            dense_units = hp.Int('denseUnits_' + str(i),
                                 configDense["minDenseUnits"],
                                 configDense["maxDenseUnits"],
                                 configDense["stepDenseUnits"])
            model.add(Dense(units=dense_units, activation='relu'))
        model.add(Dense(self.nOutput))

        configTrain = self.config["model"]["lstm"]["train"]
        if configTrain["optimizer"] == "adam":
            optimizer = Adam(hp.Choice('learningRate', configTrain["learningRates"]))
        elif configTrain["optimizer"] == "sgd":
            optimizer = SGD(hp.Choice('learningRate', configTrain["learningRates"]))
        elif configTrain["optimizer"] == "rmsprop":
            optimizer = RMSprop(hp.Choice('learningRate', configTrain["learningRates"]))
        model.compile(optimizer=optimizer,
                      loss=configTrain["loss"],
                      metrics=configTrain["metrics"])
        return model

    
    def fit(self, hp, model, x, y, **kwargs):

        config = self.config["model"]["lstm"]["train"]
        
        # Add EarlyStopping callback
        callbacks = kwargs.pop('callbacks', [])
        callbacks.append(EarlyStopping(monitor=config["earlyStopping"]["monitor"],
                                       patience=config["earlyStopping"]["patience"],
                                       restore_best_weights=True))
        kwargs['callbacks'] = callbacks

        # Add batch size
        kwargs['batch_size'] = hp.Int('batch_size',
                                      config["batch"]["minBatchSize"],
                                      config["batch"]["maxBatchSize"],
                                      config["batch"]["stepBatchSize"])

        return model.fit(x, y, **kwargs)
            


# # Define LSTM model
# model = Sequential()
# model.add(LSTM(200, activation='relu', input_shape=(nInput, xTrain.shape[1])))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(3))


# # Compile the model
# model.compile(optimizer='adam', loss='mse')

# # Fit the model
# history = model.fit(xTrainSeq, predMatrix.yTrain[nInput-1::stepSize], epochs=100, batch_size=32, validation_data=(xTestSeq, predMatrix.yTest[nInput-1::stepSize]))