# K-means
# time complexity: O(k x n x m x t)
# k: number of clusters 
# n: number of data points
# m: number of features (dimensions) of each data point 
# t: number of iterations needed for convergence

import numpy as np
from sklearn.base import BaseEstimator

class KMeans(BaseEstimator):
    def __init__(self, n_clusters, max_iter=100, random_seed=None, verbose=False) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
        self.verbose = verbose

    
    def fit(self, X):
        #Randomly select the initial centroids from the given points (with no replacements)
        idx = self.random_state.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        if self.verbose:
            print('Centroids:', self.centroids)

        # Allocate a distances matrix between the data points and the centroids
        distances = np.zeros((len(X), self.n_clusters))
        prev_labels = None

        for iteration in range(self.max_iter):
            if self.verbose:
                print('\nIteration', iteration)
            
            # Compute the distance to the cluster centroids
            for i in range(self.n_clusters):
                distances[:, i] = np.sum((X - self.centroids[i])**2, axis=1)
            
            # Assign each datapoint to the cluster with the nearest centroids
            self.labels = np.argmin(distances, axis=1)
            if self.verbose: print('Label:', self.labels)

            #Check if there was no change in the cluster assignment
            if np.all(self.labels == prev_labels):
                break 
            prev_labels = self.labels

            #Recompute the centroids
            for i in range(self.n_clusters):
                self.centroids[i] = np.mean(X[self.labels == i], axis=0)

                # Handle empty clusters
                if np.isnan(self.centroids[i]).any():
                    self.centroids[i] = X[self.random_state.choice(len(X))]
            if self.verbose: print('Centroids:', self centroids)
            