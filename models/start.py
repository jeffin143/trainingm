#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:36:41 2018

@author: jeffin
"""

from train import stratified_cv
from train import show_cm
from train import plot_cm
from train import plot_roc
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import  AdaBoostClassifier
import os
import matplotlib.pylab as pylab
import time
import pandas as pd
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #To suppress warnings about CPU instruction set
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import naive_bayes
import xgboost

#import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from sklearn import cross_validation

def take(labels,X,y):
        
    
    
    print("Classifier\t Execution Time \t Accuracy")
    GNB=stratified_cv(X, y, naive_bayes.GaussianNB,"Gaussian NB ") 
    RFC=stratified_cv(X, y, ensemble.RandomForestClassifier, "Random Forest Classifier",max_depth=12)
    KNC=stratified_cv(X, y, neighbors.KNeighborsClassifier,"K Neighbors Classifier") 
    SVM=stratified_cv(X, y, svm.SVC,"Support Vector Machine",max_iter=5000)
    DTC=stratified_cv(X, y, tree.DecisionTreeClassifier, "Decision Tree")
    XGB=stratified_cv(X,y,xgboost.XGBClassifier,"XGBoost")
    MLP=stratified_cv(X,y,neural_network.MLPClassifier,"MLP NN")
    ADB=stratified_cv(X,y,AdaBoostClassifier,"ADA BOOST")
    QA=stratified_cv(X,y,QuadraticDiscriminantAnalysis,"QDA DISCRI")


    #Area under ROC curve
    plot_roc(y,GNB)
    plot_roc(y,RFC)
    plot_roc(y,KNC)
    plot_roc(y,SVM)
    plot_roc(y,DTC)
    plot_roc(y,XGB)  
    plot_roc(y,MLP)
    plot_roc(y,ADB)
    plot_roc(y,QA)
    
    #Plot Confusion Matrix
      
    plot_cm(y, GNB, labels)
    plot_cm(y, RFC, labels)
    plot_cm(y, KNC, labels)
    plot_cm(y, SVM, labels)
    plot_cm(y, DTC, labels)
    plot_cm(y, XGB, labels)
    plot_cm(y, MLP, labels)
    plot_cm(y, ADB, labels)
    plot_cm(y, QA, labels)
    
    plot_clf_cmpr([GNB,RFC,KNC,SVM,DTC,XGB,MLP,ADB,QA])
    
    
    #printing classification report
    from sklearn.metrics import classification_report
    print('Naive Bayes Classifier:\n {}\n'.format(classification_report(y, GNB['y_pred'])))
    print('Random Forest Classifier:\n {}\n'.format(classification_report(y, RFC['y_pred'])))
    print('K Nearest Neighbor Classifier:\n {}\n'.format(classification_report(y, KNC['y_pred'])))
    print('"Decision Tree"\n {}\n'.format(classification_report(y, DTC['y_pred'])))
    print('XGBoost  \n {}\n'.format(classification_report(y, XGB['y_pred'])))
    print('MLP \n {}\n'.format(classification_report(y, MLP['y_pred'])))
    print('ADA boost :\n {}\n'.format(classification_report(y, ADB['y_pred'])))
    print('qadratic discriminant :\n {}\n'.format(classification_report(y, QA['y_pred'])))
