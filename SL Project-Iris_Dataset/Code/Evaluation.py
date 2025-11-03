# -*- coding: utf-8 -*-
"""
Created on Tue May  9 20:24:17 2023

@author: jasmi
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc , roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics

iris = datasets.load_iris()
acc=[]
pr =[]
recall=[]
youdin=[]

X = iris['data']
Y = iris['target']
Y1 = iris['target']

one_hot_encoder = OneHotEncoder(sparse_output=False)
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
X_train,x_val,y_train,y_val = train_test_split(X_train,y_train, train_size =0.75)

X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y1, test_size=0.15)

#knn
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(X1_train,y1_train)
k_prob= KNN.predict_proba(X_test)
k_pred= KNN.predict(X1_test)
k_acc= metrics.accuracy_score(y1_test, k_pred)
r_acc= metrics.recall_score(y1_test, k_pred, average="weighted")
p_acc= metrics.precision_score(y1_test, k_pred,average="weighted")
pr.append(p_acc)
recall.append(r_acc)
acc.append(k_acc)
fpr, tpr,threshold = roc_curve(y_test[:,1], k_prob[:,1])
youdin.append(max(np.subtract(tpr,fpr)))

#neural network
MLP = MLPClassifier(hidden_layer_sizes=(10,10), max_iter =1000)
MLP.fit(X_train, y_train)

MLP_pred = MLP.predict(X_test)
MLP_prob = MLP.predict_proba(X_test)
nn_acc= metrics.accuracy_score(y_test, MLP_pred)
acc.append(nn_acc)
r_acc= metrics.recall_score(y_test, MLP_pred,average="weighted")
p_acc= metrics.precision_score(y_test, MLP_pred,average="weighted")
pr.append(p_acc)
recall.append(r_acc)
fpr, tpr,threshold = roc_curve(y_test[:,1], MLP_prob[:,1])
youdin.append(max(np.subtract(tpr,fpr)))

#Decision tree
DTC1 = DecisionTreeClassifier(max_depth =5, random_state=0, criterion="gini")
DTC1.fit(X_train, y_train)
DTC1_pred = DTC1.predict(X_test)
DTC1_prob = DTC1.predict_proba(X_test)

dtc_acc= metrics.accuracy_score(y_test, DTC1_pred)
acc.append(dtc_acc)
r_acc= metrics.recall_score(y_test, DTC1_pred,average="weighted")
p_acc= metrics.precision_score(y_test, DTC1_pred,average="weighted")
pr.append(p_acc)
recall.append(r_acc)
fpr, tpr,threshold = roc_curve(y_test[:,1], DTC1_pred[:,1])
youdin.append(max(np.subtract(tpr,fpr)))

#Create a svm Classifier
clf = svm.SVC(kernel='linear',probability=True) # Linear Kernel

#Train the model using the training sets
clf.fit(X1_train, y1_train)

#Predict the response for test dataset
y_pred = clf.predict(X1_test)
y_prob= clf.predict_proba(X_test)
svm_acc= metrics.accuracy_score(y1_test, y_pred)
acc.append(svm_acc)
r_acc= metrics.recall_score(y1_test, y_pred,average="weighted")
p_acc= metrics.precision_score(y1_test, y_pred,average="weighted")
pr.append(p_acc)
recall.append(r_acc)
fpr, tpr,threshold = roc_curve(y_test[:,1], y_prob[:,1])
youdin.append(max(np.subtract(tpr,fpr)))


l = ["KNN", "NN"," DT", "SVM"]

for i in l:
    plt.subplot(2, 2, 1)
    plt.bar(l,acc)
    plt.xlabel("models")
    plt.ylabel("Accuracy")
    
    plt.subplot(2, 2, 2)
    plt.bar(l,pr)
    plt.xlabel("models")
    plt.ylabel("Precision")
    
    plt.subplot(2, 2, 3)
    plt.bar(l,recall)
    plt.xlabel("models")
    plt.ylabel("Recall")
    
    plt.subplot(2, 2, 4)
    plt.bar(l,youdin)
    plt.xlabel("models")
    plt.ylabel("Youdin")


    





