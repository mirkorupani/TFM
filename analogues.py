import json
from sklearn.cluster import KMeans, SpectralClustering
from mdapy import max_diss_alg
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import numpy as np
import pandas as pd

class Analogues():
    """Class to apply the analogues method to the prediction matrix"""


    def __init__(self, configFilePath, predictionMatrix):
        
        with open(configFilePath) as f:
            self.config = json.load(f)
        self.predMatrix = predictionMatrix
        self.clustering = self.getClustering()
        self.centroids, self.xAnalogues, self.yAnalogues = self.getAnalogues()
        self.classifier = self.getClassifier()


    def getClustering(self):
        """Gets a clustering model
        :return: sklearn.neighbors.NearestNeighbors, clustering"""

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
        """Gets the analogues"""

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
    

    def getClassifier(self):
        """Gets the classifier
        :return: sklearn.neighbors.KNeighborsClassifier, classifier"""

        classifier = KNeighborsClassifier(n_neighbors=self.config["model"]["analogues"]["nNeighbors"], weights=self.config["model"]["analogues"]["weights"])

        if self.config["model"]["analogues"]["clustering"] == "kMeans":
            return classifier.fit(self.clustering.cluster_centers_, self.centroids)
        elif self.config["model"]["analogues"]["clustering"] == "maxDiss":
            return classifier.fit(self.clustering[0], self.centroids)
        elif self.config["model"]["analogues"]["clustering"] == "spectral":
            return classifier.fit(self.predMatrix.xTrain, self.clustering.labels_)
        

    def predict(self, x):
        """Predicts the analogues
        :param x: pandas.DataFrame or numpy.ndarray, predictors
        :return: pandas.DataFrame or numpy.ndarray, analogues prediction
        """

        probMatrix = self.classifier.predict_proba(x)
        if isinstance(x, np.ndarray):
            prediction = np.empty((x.shape[0], self.yAnalogues.shape[1]))
            for i in range(prediction.shape[0]):
                prediction[i] = np.dot(probMatrix[i], self.yAnalogues)
        elif isinstance(x, pd.DataFrame):
            prediction = pd.DataFrame(index=x.index, columns=self.yAnalogues.columns)
            for i in range(prediction.shape[0]):
                prediction.iloc[i] = np.dot(probMatrix[i], self.yAnalogues)
        
        return prediction