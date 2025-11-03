# -*- coding: utf-8 -*-
"""
Created on Sat May 5 12:35:56 2023

@author: jasmi
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from KNeighborsClassifier import KNeighbors #my implementation
from NeuralNetworkClassifier import NeuralNetworkClassifier #my implementation
from sklearn.metrics import auc , roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Unpack the iris dataset, from UCI Machine Learning Repository
iris = datasets.load_iris()

X = iris['data']
Y = iris['target']
one_hot_encoder = OneHotEncoder(sparse_output=False) #need this for NN

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
X_train,x_val,y_train,y_val = train_test_split(X_train,y_train, train_size =0.75)

thresholds = np.arange(0, 1.001,0.002) # 1001 probability thresholds

def most_common(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

def minkowski(point, data, p=2):
    return np.sum(np.abs(point - data)**p, axis=1)**(1/p)

def manhattan(point, data):
    return sum(abs(val1-val2) for val1, val2 in zip(point,data))

def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0) #used deterministic prediction
        
        #calculate fp,tp,fn,tn
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]

if __name__ == "__main__":
    
    '''Performance of k'''
    accuracies = []
    y_pred = []
    ks = range(1, 30)
    for k in ks:
        knn = KNeighbors(k=k)
        knn.fit(X_train, y_train)
        y_p,accuracy = knn.evaluate(X_test, y_test)
        accuracies.append(accuracy)
        y_pred.append(y_p)
        
    plt.plot(ks, accuracies)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Performance of knn")
    plt.show()
    maxk= np.argmax(accuracies)
    print("Classification Report:", classification_report(y_test, y_pred[maxk]))
    print("Confusion matrix of KNN", confusion_matrix(y_test,y_pred[maxk]))
    
    ax= plt.subplot(2,2,1)
    sns.heatmap(confusion_matrix(y_test,y_pred[maxk]), annot=True,ax=ax)

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 

    # fig.xlabel('Predicted labels');fig.ylabel('True labels'); 
    # fig.title('Confusion Matrix'); 
    
    '''Implementing NN'''
    #Binarize the output so it can learn and give probablistic output
    
    Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))
    y_train = one_hot_encoder.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = one_hot_encoder.fit_transform(np.array(y_test).reshape(-1, 1))
    y_val= one_hot_encoder.fit_transform(np.array(y_val).reshape(-1, 1))
    f = len(X[0]) # Number of features
    o = len(Y[0]) # Number of outputs / classes

    Onelayers = [f, 10, o] # Number of nodes in layers
    Twolayers = [f, 10,10, o] # Number of nodes in layers
    
    fpr ={}
    tpr ={}
    youden={}

    lr, epochs = 0.15, 100
    counter=0
    layers = [Onelayers,Twolayers]
    for layer in layers:
        NN = NeuralNetworkClassifier(layer)
        weights = NN.fit(X_train, y_train, x_val, y_val, epochs=epochs, nodes=layer, lr=lr)
        accuracy,output,err = NN.Accuracy(X_test, y_test, weights)
        print("Testing Accuracy: {}".format(accuracy))
        print("0-1loss:", err)
        output = np.array(output)
        fpr[counter], tpr[counter] = roc_curve(y_test[:,1],output[:,1],thresholds)  #for verscicolor, use 0 for setsosa, 2 for virginica
        youden[counter] = np.subtract(tpr[counter],fpr[counter])
        counter+=1
        
    plt.subplot(2, 2, 1)
    plt.plot([0,1], [0,1], linestyle='--', label='Random predictor')
    plt.plot(fpr[0], tpr[0])
    plt.title("One 10 units hidden layer")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    print("Youden for One 10 units hidden layer:", max(youden[0]))
    
    
    plt.subplot(2, 2, 2)
    plt.plot([0,1], [0,1], linestyle='--', label='Random predictor')
    plt.plot(fpr[1], tpr[1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title("Two 10 units hidden layer")
    print("Youden for two 10 units hidden layer:", max(youden[1]))
    
    plt.subplot(2, 2, 3)
    plt.plot([0,1], [0,1], linestyle='--', label='Random predictor') #ROC one v/s rest
    for i in range(3):    
        fpr, tpr = roc_curve(y_test[:,i],output[:,i],thresholds)
        plt.plot(fpr, tpr, label = i)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
    
    plt.show()
    
    
    
    