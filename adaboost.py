from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import json
import numpy as np

class AdaBoost():
    """Class to train an AdaBoost model from the prediction matrix"""


    def __init__(self, config, predMatrix):
        """
        :param config: str, path to the configuration file
        :param predMatrix: PredictionMatrix, prediction matrix
        """
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
                self.config = json.load(f)
        self.predMatrix = predMatrix
        self.model = self.trainModel()
    

    def ksStatistic(self, data, dataRecon):
        """Calculates the Kolmogorov-Smirnov (KS) statistic between two samples.
        
        :param data: np.array, original data
        :param dataRecon: np.array, reconstructed data
        
        :return: float, KS statistic
        """
        xo = np.asarray(data).flatten()
        xs = np.asarray(dataRecon).flatten()
        xo_sorted = np.sort(xo)
        xs_sorted = np.sort(xs)
        return np.max(np.abs(xo_sorted - xs_sorted))


    def ksScorer(self, estimator, X, y):
        """Scorer function to calculate KS statistic during grid search.
        
        :param estimator: sklearn model, model to evaluate
        :param X: np.array, features
        :param y: np.array, target values
        
        :return: float, KS statistic
        """
        yPred = estimator.predict(X)
        return -self.ksStatistic(y, yPred)
    

    def trainModel(self):
        """
        Trains the AdaBoost model
        
        :return: dict, trained AdaBoost models
        """
        
        # Create a time series split cross-validator
        tscv = TimeSeriesSplit(n_splits=self.config["model"]["adaBoost"]["nSplits"])
        
        # Set the parameters for the model
        paramGrid = {
            "estimator__max_depth": self.config["model"]["adaBoost"]["estimator"]["maxDepth"],
            "estimator__criterion": self.config["model"]["adaBoost"]["estimator"]["criterion"],
            "estimator__splitter": self.config["model"]["adaBoost"]["estimator"]["splitter"],
            "estimator__min_samples_split": self.config["model"]["adaBoost"]["estimator"]["minSamplesSplit"],
            "estimator__min_samples_leaf": self.config["model"]["adaBoost"]["estimator"]["minSamplesLeaf"],
            "n_estimators": self.config["model"]["adaBoost"]["nEstimators"],
            "learning_rate": self.config["model"]["adaBoost"]["learningRate"],
            "loss": self.config["model"]["adaBoost"]["loss"]
        }
        # Train an AdaBoost regressor for each predictand
        model = {}
        for var in self.predMatrix.y.columns:
            
            # Train the model
            scoring = self.ksScorer if self.config["model"]["adaBoost"]["scoring"] == "ks" else self.config["model"]["adaBoost"]["scoring"]
            model[var] = GridSearchCV(AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=self.config["randomState"]), paramGrid, cv=tscv, scoring=scoring, n_jobs=self.config["model"]["adaBoost"]["nJobs"])
            model[var].fit(self.predMatrix.xTrain, self.predMatrix.yTrain[var])

            # Print the best parameters and best score
            print(f"Best parameters for {var}: {model[var].best_params_}")
            print(f"Best score for {var}: {model[var].best_score_}")

        return model
    

