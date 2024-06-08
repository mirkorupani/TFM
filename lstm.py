# from predictors import Predictors
# from predictands import Predictands
# from predictionMatrix import PredictionMatrix
# import matplotlib.pyplot as plt
# from plotFunct import combinedPlot
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
import keras_tuner as kt
from keras import backend as K
from keras.metrics import Metric
import pandas as pd
import numpy as np
import json
import os


class RnnLstm():
    """Class to train an LSTM model from the prediction matrix"""

    def __init__(self, config, predMatrix, modelName=None, overwrite=False, hyperparameters=None, nOutput=None):
        """
        :param config: str, path to the configuration file or dictionary with the configuration
        :param predMatrix: PredictionMatrix, prediction matrix
        :param modelName: str, name of the model
        :param overwrite: bool, whether to overwrite the model
        :param hyperparameters: dict, hyperparameters for the model
        :param nOutput: int, number of output variables
        
        :return: None
        """
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
                self.config = json.load(f)
        self.predMatrix = predMatrix
        self.nTimesteps = self.config["model"]["lstm"]["nTimesteps"]
        self.stepSize = self.config["model"]["lstm"]["stepSize"]
        self.nOutput = len(self.config["predictands"]["variables"]) if nOutput is None else nOutput
        self.xTrainSeq, self.xTestSeq = self.preprocessData()
        self.model, self.bestHP = self.trainModel(modelName=modelName, overwrite=overwrite, hyperparameters=hyperparameters)
    

    def preprocessData(self, newData=False):
        """
        Preprocesses the data to be fed into the LSTM model
        
        :param newData: pd.DataFrame, new data to preprocess
        
        :return: np.array, input sequences
        """
        if newData is not False:
            xTrain = pd.DataFrame(newData, index=newData.index)
        else:
            # Convert xTrain and xTest to a pandas DataFrame
            xTrain = pd.DataFrame(self.predMatrix.xTrain, index=self.predMatrix.yTrain.index)
            xTest = pd.DataFrame(self.predMatrix.xTest, index=self.predMatrix.yTest.index)
        
        # If the historical file is a list of files and the time steps are NOT consecutive
        # NOTE: if some periods are consecutive and some are not, the code will not work
        if isinstance(self.config["predictands"]["hisFile"], list) and len(np.unique(np.diff(xTrain.index))) != 1 and newData is False:
            xTrainSeq = np.empty((len(range(0, len(xTrain) - self.nTimesteps + 1, self.stepSize)), self.nTimesteps, xTrain.shape[1]))
                                 
            # Create the input sequences
            minTimeDiff = np.min(np.unique(np.diff(xTrain.index)))
            timeDiff = minTimeDiff
            i = 0
            for period in range(len(self.config["predictands"]["hisFile"])):
                while timeDiff == minTimeDiff:
                    xTrainSeq[i] = xTrain.iloc[(period * self.nTimesteps)+i:(period * self.nTimesteps)+i+self.nTimesteps].values
                    i += 1
                    timeDiff = xTrain.index[i] - xTrain.index[i - 1]
        else:
            xTrainSeq = np.empty((len(range(0, len(xTrain) - self.nTimesteps + 1, self.stepSize)), self.nTimesteps, xTrain.shape[1]))

            # Create the input sequences
            for i, idx in enumerate(range(0, len(xTrain) - self.nTimesteps + 1, self.stepSize)):
                xTrainSeq[i] = xTrain.iloc[idx:idx+self.nTimesteps].values

        if newData is False:
            # Create an empty array to store the input sequences
            xTestSeq = np.empty((len(range(0, len(xTest) - self.nTimesteps + 1, self.stepSize)), self.nTimesteps, xTest.shape[1]))

            # Create the input sequences
            for i, idx in enumerate(range(0, len(xTest) - self.nTimesteps + 1, self.stepSize)):
                xTestSeq[i] = xTest.iloc[idx:idx+self.nTimesteps].values
        
        if newData is not False:
            return xTrainSeq
        else:
            return xTrainSeq, xTestSeq
    

    def trainModel(self, overwrite=False, modelName=None, hyperparameters=None):
        """
        Trains the LSTM model
        
        :param overwrite: bool, whether to overwrite the model
        :param modelName: str, name of the model
        :param hyperparameters: dict, hyperparameters for the model
        
        :return: keras model, trained LSTM model
        """
        modelName = f"sta{self.config['predictands']['station']}Model" if modelName is None else modelName
        modelPath = "results\\lstm\\" + modelName + ".h5"

        if not overwrite and os.path.exists(modelPath):
            print("Model loaded from disk.")
            return load_model(modelPath), None
        
        configHyperband = self.config["model"]["lstm"]["hyperband"]
        
        if hyperparameters:
            bestModel = self.buildModelWithHyperparameters(hyperparameters)
            bestHP = hyperparameters  # Store the provided hyperparameters for reference
        else:
            inputShape = (self.nTimesteps, self.predMatrix.xTrain.shape[1])
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
                project_name=configHyperband["projectName"] + modelName,
                seed=self.config["randomState"]
            )

            # Search for the best hyperparameters
            if isinstance(self.config["predictands"]["hisFile"], list) and len(np.unique(np.diff(self.predMatrix.yTrain.index))) != 1:
                # Get the labels (yTrain) considering multiple historical files
                timeDiff = np.diff(self.predMatrix.yTrain.index)
                idx = np.where(timeDiff != np.min(np.unique(timeDiff)))[0]
                yTrain = np.empty((len(self.xTrainSeq), self.nOutput))
                for i in range(len(idx)):
                    if i == 0:
                        yTrain[:idx[i]-self.nTimesteps+1] = self.predMatrix.yTrain[self.nTimesteps-1:idx[i]]
                    else:
                        yTrain[idx[i-1]-i*(self.nTimesteps-1):idx[i]-i*(self.nTimesteps-1)] = self.predMatrix.yTrain[idx[i-1]+i*(self.nTimesteps-1):idx[i]+i*(self.nTimesteps-1)]
                yTrain[idx[-1]-len(idx)*(self.nTimesteps-1):] = self.predMatrix.yTrain[idx[-1]+len(idx)*(self.nTimesteps-1):]
                tuner.search(self.xTrainSeq, yTrain, validation_data=(self.xTestSeq, self.predMatrix.yTest[self.nTimesteps-1::self.stepSize]))
            else:
                tuner.search(self.xTrainSeq, self.predMatrix.yTrain[self.nTimesteps-1::self.stepSize], validation_data=(self.xTestSeq, self.predMatrix.yTest[self.nTimesteps-1::self.stepSize]))

            # Get the best model
            bestHP = tuner.get_best_hyperparameters()[0]
            bestModel = model.build(bestHP)
        
        # Fit the best model
        if isinstance(self.config["predictands"]["hisFile"], list) and len(np.unique(np.diff(self.predMatrix.yTrain.index))) != 1:
            bestModel.fit(
                self.xTrainSeq,
                yTrain,
                validation_data=(self.xTestSeq, self.predMatrix.yTest[self.nTimesteps-1::self.stepSize]),
                epochs=configHyperband["maxEpochs"],
                callbacks=[EarlyStopping(
                    monitor=self.config["model"]["lstm"]["train"]["earlyStopping"]["monitor"],
                    patience=self.config["model"]["lstm"]["train"]["earlyStopping"]["patience"],
                    restore_best_weights=True
                    )],
                batch_size=bestHP["values"]["batch_size"]
            )
        else:
            configEarlyStopping = self.config["model"]["lstm"]["train"]["earlyStopping"]
            bestModel.fit(
                self.xTrainSeq,
                self.predMatrix.yTrain[self.nTimesteps-1::self.stepSize],
                validation_data=(self.xTestSeq, self.predMatrix.yTest[self.nTimesteps-1::self.stepSize]),
                epochs=configHyperband["maxEpochs"],
                callbacks=[EarlyStopping(
                    monitor=configEarlyStopping["monitor"],
                    patience=configEarlyStopping["patience"],
                    restore_best_weights=True
                    )],
                batch_size=bestHP["values"]["batch_size"]
            )
        
        # Save the model
        bestModel.save(modelPath)
        print("Model saved to disk.")

        return bestModel, bestHP
    

    def buildModelWithHyperparameters(self, hyperparameters):
        """
        Builds the LSTM model using the provided hyperparameters

        :param hyperparameters: dict, hyperparameters for the model

        :return: keras model, LSTM model
        """
        model = Sequential()

        # Get LSTM layer configurations from hyperparameters
        num_lstm_layers = hyperparameters['values']['numLstmLayers']
        for i in range(num_lstm_layers):
            return_sequences = i < num_lstm_layers - 1  # Set return_sequences to True for all except last LSTM layer
            lstm_units = hyperparameters['values'][f'lstmUnits_{i}']
            dropout_lstm = hyperparameters['values'][f'dropoutLstm_{i}']
            dropout_rate_lstm = hyperparameters['values'][f'dropoutRateLstm_{i}']
            
            if i == 0:
                model.add(LSTM(units=lstm_units,
                            activation='relu',
                            input_shape=(self.nTimesteps, self.predMatrix.xTrain.shape[1]),
                            return_sequences=return_sequences))
            else:
                model.add(LSTM(units=lstm_units,
                            activation='relu',
                            return_sequences=return_sequences))
            
            model.add(BatchNormalization())
            
            if dropout_lstm:
                model.add(Dropout(rate=dropout_rate_lstm))

        # Get Dense layer configurations from hyperparameters
        num_dense_layers = hyperparameters['values']['numDenseLayers']
        for i in range(num_dense_layers):
            dense_units = hyperparameters['values'][f'denseUnits_{i}']
            model.add(Dense(units=dense_units, activation='relu'))
        
        model.add(Dense(self.nOutput))

        # Get training configurations from hyperparameters
        optimizer_name = self.config["model"]["lstm"]["train"]["optimizer"]
        learning_rate = hyperparameters['values']['learningRate']
        
        if optimizer_name == "adam":
            optimizer = Adam(learning_rate)
        elif optimizer_name == "sgd":
            optimizer = SGD(learning_rate, momentum=0.9)
        elif optimizer_name == "rmsprop":
            optimizer = RMSprop(learning_rate)

        # Compile the model
        if self.config["model"]["lstm"]["hyperband"]["objective"] == "val_ks_statistic":
            model.compile(optimizer=optimizer,
                        loss=self.config["model"]["lstm"]["train"]["loss"],
                        metrics=[KSMetric()])
        else:
            model.compile(optimizer=optimizer,
                        loss=self.config["model"]["lstm"]["train"]["loss"],
                        metrics=self.config["model"]["lstm"]["train"]["metrics"])
        
        return model



# Create hyperregressor class
class HyperRegressor(kt.HyperModel):
    """HyperModel class for LSTM model"""
        
    def __init__(self, inputShape, nOutput, config):
        """
        :param inputShape: tuple, input shape of the model
        :param nOutput: int, number of output variables
        :param config: dict, configuration file
        
        :return: None
        """
        self.inputShape = inputShape
        self.nOutput = nOutput
        self.config = config
    

    def build(self, hp):
        """
        Builds the LSTM model
        
        :param hp: HyperParameters, hyperparameters for the model
        
        :return: keras model, LSTM model
        """
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
            optimizer = SGD(hp.Choice('learningRate', configTrain["learningRates"]), momentum=0.9)
        elif configTrain["optimizer"] == "rmsprop":
            optimizer = RMSprop(hp.Choice('learningRate', configTrain["learningRates"]))
        if self.config["model"]["lstm"]["hyperband"]["objective"] == "val_ks_statistic":
            model.compile(optimizer=optimizer,
                          loss=configTrain["loss"],
                          metrics=[KSMetric()])
        else:
            model.compile(optimizer=optimizer,
                        loss=configTrain["loss"],
                        metrics=configTrain["metrics"])
        return model

    
    def fit(self, hp, model, x, y, **kwargs):
        """
        Fits the LSTM model
        
        :param hp: HyperParameters, hyperparameters for the model
        :param model: keras model, LSTM model
        :param x: np.array, input sequences
        :param y: np.array, target values
        :param kwargs: dict, additional arguments
        
        :return: keras history, training history
        """

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


class KSMetric(Metric):
    """Custom metric to calculate the KS statistic"""

    def __init__(self, name='ks_statistic', **kwargs):
        """
        :param name: str, name of the metric
        
        :return: None
        """
        super(KSMetric, self).__init__(name=name, **kwargs)
        self.ks_statistic = self.add_weight(name='ks_statistic', initializer='zeros')
        self.samples = self.add_weight(name='samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        ks = K.max(K.abs(K.cumsum(y_true / K.sum(y_true)) - K.cumsum(y_pred / K.sum(y_pred))))
        self.ks_statistic.assign_add(ks)
        self.samples.assign_add(1)

    def result(self):
        return self.ks_statistic / self.samples

    def reset_states(self):
        self.ks_statistic.assign(0)
        self.samples.assign(0)