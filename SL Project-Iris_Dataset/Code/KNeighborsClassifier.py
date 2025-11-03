# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:16:48 2023

@author: jasmi
"""
import numpy as np
from math import *

def most_common(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

def minkowski(point, data, p=5):
    return np.sum(np.abs(point - data)**p, axis=1)**(1/p)

def manhattan(point, data):
    return sum(abs(val1-val2) for val1, val2 in zip(point,data))
 

class KNeighbors():
    
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        
        return list(map(most_common, neighbors))
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return [y_pred,accuracy]