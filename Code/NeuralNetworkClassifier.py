# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:13:18 2023

@author: jasmi

"""

import numpy as np

class NeuralNetworkClassifier():
    def __init__(self,units):
        self.units =units
        self.lr= 0.15
        
    def fit(self,X_train, y_train, X_val=None, Y_val=None, epochs=10, nodes=[], lr=0.15 ):
        hidden_layers = len(nodes) - 1
        weights = self.InitializeWeights(nodes)

        for epoch in range(1, epochs+1):
            weights = self.Train(X_train, y_train, lr, weights)
            #episodes.append(epoch)

            if(epoch % 20 == 0):
                print("Epoch {}".format(epoch))
                acc,y_pred,err = self.Accuracy(X_train, y_train, weights)
                print("Training Accuracy:{}".format(acc,err))
                if X_val.any():
                    accval,yval_pred,valerr = self.Accuracy(X_val, Y_val, weights)
                    print("Validation Accuracy:{}".format(accval,valerr))
            
        return weights
    
    def InitializeWeights(self,nodes):
        """Initialize weights with random values in [-1, 1] (including bias)"""
        layers, weights = len(nodes), []
    
        for i in range(1, layers):
            w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
                 for j in range(nodes[i])]
            weights.append(np.matrix(w))
    
        return weights
    
    
    def ForwardPropagation(self,x, weights, layers):
        activations, layer_input = [x], x
        for j in range(layers):
            activation = self.sig(np.dot(layer_input, weights[j].T))
            activations.append(activation)
            layer_input = np.append(1, activation) # Augment with bias
    
        return activations
    
    def BackPropagation(self,y, activations, weights, layers):
        outputFinal = activations[-1]
        error = np.matrix(y - outputFinal) # Error at output
    
        for j in range(layers, 0, -1):
            currActivation = activations[j]
        
            if(j > 1):
                # Augment previous activation
                prevActivation = np.append(1, activations[j-1])
            else:
                # First hidden layer, prevActivation is input (without bias)
                prevActivation = activations[0]
        
            delta = np.multiply(error, self.sigDerivative(currActivation))
            weights[j-1] += self.lr * np.multiply(delta.T, prevActivation)

            w = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
            error = np.dot(delta, w) # Calculate error for current layer
            
    
        return weights
    
    def Train(self,X, Y, lr, weights):
        layers = len(weights)
        for i in range(len(X)):
            x, y = X[i], Y[i]
            x = np.matrix(np.append(1, x)) # Augment feature vector
        
            activations = self.ForwardPropagation(x, weights, layers)
            weights = self.BackPropagation(y, activations, weights, layers)

        return weights
    
    def Predict(self,item, weights):
        layers = len(weights)
        item = np.append(1, item) # Augment feature vector
    
        ##_Forward Propagation_##
        activations = self.ForwardPropagation(item, weights, layers)
    
        outputFinal = activations[-1].A1
        index = self.FindMaxActivation(outputFinal)

        # Initialize prediction vector to zeros
        y = [0 for i in range(len(outputFinal))]
        y[index] = 1  # Set guessed class to 1

        return [y,outputFinal] # Return prediction vector
    
    def FindMaxActivation(self,output):
        """Find max activation in output"""
        m, index = output[0], 0
        for i in range(1, len(output)):
            if(output[i] > m):
                m, index = output[i], i
    
        return index
    
    def Accuracy(self,X, Y, weights):
        """Run set through network, find overall accuracy"""
        correct = 0
        outputs =[]
        for i in range(len(X)):
            x, y = X[i], list(Y[i])
            guess,y_pred = self.Predict(x, weights)
            outputs.append(y_pred)

            if(y == guess):
                # Guessed correctly
                correct += 1
                
        acc = correct / len(X)
        err = 1-acc

        return correct / len(X),outputs,err
    
    def sig(self,x):
        return 1/(1 + np.exp(-x))
    
    def sigDerivative(self,x):
        return np.multiply(x, 1-x)