import json
from sklearn.cluster import KMeans, SpectralClustering
from mdapy import max_diss_alg
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import pandas as pd

class Analogues():
    """Class to apply the analogues method to the prediction matrix"""


    def __init__(self, config, predictionMatrix):
        """
        Initializes the Analogues class

        :param config: str or dict, path to the configuration file or configuration dictionary
        :param predictionMatrix: PredictionMatrix, prediction matrix

        :return: None
        """
        
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as f:
                self.config = json.load(f)
        self.predMatrix = predictionMatrix
        self.clustering = self.getClustering()
        self.centroids, self.xAnalogues, self.yAnalogues = self.getAnalogues()
        self.regressor = self.getRegressor()


    def getClustering(self):
        """
        Gets the clustering algorithm
        
        :return: fitted clustering algorithm (KMeans, max_diss_alg, SpectralClustering)
        """

        randomState = self.config["randomState"]

        nClusters = self.config["model"]["analogues"]["nAnalogues"]

        if self.config["model"]["analogues"]["clustering"] == "kMeans":
            return KMeans(n_clusters=nClusters, random_state=randomState).fit(self.predMatrix.xTrain)
        
        elif self.config["model"]["analogues"]["clustering"] == "maxDiss":
            if isinstance(self.predMatrix.xTrain, pd.DataFrame):
                dataTuple = tuple([self.predMatrix.xTrain[col] for col in self.predMatrix.xTrain.columns])
            elif isinstance(self.predMatrix.xTrain, np.ndarray):
                dataTuple = tuple([self.predMatrix.xTrain[:, i] for i in range(self.predMatrix.xTrain.shape[1])])
            return max_diss_alg(dataTuple, nClusters, seed_index=randomState)
        
        elif self.config["model"]["analogues"]["clustering"] == "spectral":
            spectralClustering = SpectralClustering(n_clusters=nClusters, random_state=randomState)
            return spectralClustering.fit(self.predMatrix.xTrain)
    

    def getAnalogues(self):
        """
        Gets the analogues
        
        :return: tuple, (centroids, xAnalogues, yAnalogues)
        
        centroids: np.array, indices of the analogues
        xAnalogues: np.array, features of the analogues
        yAnalogues: np.array, target values of the analogues
        """

        # Get the analogues
        if self.config["model"]["analogues"]["clustering"] == "kMeans":
            kNN1 = KNeighborsClassifier(n_neighbors=1).fit(self.predMatrix.xTrain, range(len(self.predMatrix.xTrain)))
            centroids = kNN1.predict(self.clustering.cluster_centers_)
        elif self.config["model"]["analogues"]["clustering"] == "maxDiss":
            centroids = self.clustering[1]
        elif self.config["model"]["analogues"]["clustering"] == "spectral":
            centroids = np.empty(self.clustering.n_clusters, dtype=int)
            for i in range(self.clustering.n_clusters):
                clusterIndices = np.where(self.clustering.labels_ == i)[0]
                clusterDataPoints = self.predMatrix.xTrain[clusterIndices]
                distances = np.sum((clusterDataPoints[:, np.newaxis] - clusterDataPoints)**2, axis=-1)
                centroids[i] = clusterIndices[np.argmin(np.sum(distances, axis=1))]

        return centroids, self.predMatrix.xTrain[centroids], self.predMatrix.yTrain.iloc[centroids]
    

    def getRegressor(self):
        """
        Gets the regressor
        
        :return: fitted regressor (KNeighborsRegressor, KernelRidge)
        """

        regressorType = self.config["model"]["analogues"]["regressor"]
        regressorConfig = self.config["model"]["analogues"][regressorType]

        if regressorType == "knn":
            regressor = KNeighborsRegressor(**regressorConfig)

        elif self.config["model"]["analogues"]["regressor"] == "krr":
            regressor = KernelRidge(**regressorConfig)
            
        else:
            raise ValueError("Regressor not recognized")
        
        return regressor.fit(self.xAnalogues, self.yAnalogues)