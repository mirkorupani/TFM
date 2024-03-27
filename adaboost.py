from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import json

class AdaBoost():
    """Class to train an AdaBoost model from the prediction matrix"""


    def __init__(self, config, predMatrix):
        """Constructor of the class"""
        with open(config) as f:
            self.config = json.load(f)
        self.predMatrix = predMatrix
        self.model = self.trainModel()
    

    def trainModel(self):
        """Trains the AdaBoost model
        :return: dict, trained AdaBoost models"""
        
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
            model[var] = GridSearchCV(AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=self.config["randomState"]), paramGrid, cv=tscv, scoring=self.config["model"]["adaBoost"]["scoring"], n_jobs=self.config["model"]["adaBoost"]["nJobs"])
            model[var].fit(self.predMatrix.xTrain, self.predMatrix.yTrain[var])

            # Print the best parameters and best score
            print(f"Best parameters for {var}: {model[var].best_params_}")
            print(f"Best score for {var}: {model[var].best_score_}")

        return model
    

