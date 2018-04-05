#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:33:27 2018

@author: jeffin
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#from subprocess import call
from sklearn import cross_validation
import time

def stratified_cv(X, y, clf_class, clf_name, shuffle=True, n_folds=3, **kwargs):
    from sklearn.metrics import accuracy_score
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle) #it will have each folds with index of y. Use this index to get corresponding x values 
    y_pred = y.copy()
    clf = clf_class(**kwargs)
    
    start_time=time.time()
    #Iterate throught the folds. we will have ii part with 90% of train data index and jj part with 10% of test data index.
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    accuracy = accuracy_score(y_pred,y)*100
    print(str(clf_name)+"\t" +"%.3f"%(time.time()-start_time)+"\t"+ str(accuracy))
    return {'name':clf_name,'y_pred':y_pred ,'acc':accuracy}


def show_cm(y, clf, labels):
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y,clf['y_pred'])
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T) 
    print ('Confusion Matrix Stats : ',clf['name'],'\t Accuracy ',clf['acc'])
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print ("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))

def plot_cm(y, clf, labels):
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y,clf['y_pred'])
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T) 
    print ('\nConfusion Matrix Stats : '+ clf['name'])
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print ("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap=plt.cm.Blues)
    pylab.title('Confusion matrix : '+ clf['name']+' : '+str(clf['acc'])+'%\n')
    fig.colorbar(cax)
    ax.set_xticklabels([' '] + labels)
    ax.set_yticklabels(['   '] + labels)
    ax.text(0,0,percent[0][0],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(0,1,percent[0][1],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(1,0,percent[1][0],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(1,1,percent[1][1],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    pylab.xlabel('Predicted')
    pylab.ylabel('Actual')
    pylab.savefig('./ClassifierImages/binary/'+clf['name']+'.png')
    pylab.show()

    
def plot_graph(y, clf, labels):
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y,clf['y_pred'])
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T)
    print('Confusion Matrix Stats of :',clf['name'],' - ' ,clf['acc'],"%" )
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
	        print ("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))

	#Precison = true positive / (true positive + false positive)
    print ("\n\nPrecision : %.4f%%" % (percent[0][0]/(percent[0][0]+percent[0][1])*100))
    #Recall = true positive/(true positive + false negative)
    print ("Recall : %.4f%%" % (percent[0][0]/(percent[0][0]+percent[1][0])*100))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap=plt.cm.Blues)
    pylab.title('Confusion matrix of the classifier : '+clf['name']+'\n',prop={'family':'serif', 'size':'xx-small'})
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.text(0,0,percent[0][0],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(0,1,percent[0][1],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(1,0,percent[1][0],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    ax.text(1,1,percent[1][1],va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
    pylab.xlabel('Predicted')
    pylab.ylabel('True')
    pylab.savefig('ClassifierImages/binary/'+clf['name']+'_graph.png',bbox_inches='tight')
#    from matplotlib.backends.backend_pdf import PdfPages
#    pdf = PdfPages('ClassifierImages/ClassifersNew.pdf')
#    pdf.savefig(bbox_inches='tight')

def plot_roc(y_true,clf):
    from sklearn.metrics import roc_curve,auc
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, clf['y_pred'])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    sns.set('notebook', 'whitegrid', 'dark', font_scale=1.5, font='Ricty',
        rc={"lines.linewidth": 2.5, 'grid.linestyle': '--'})
    plt.figure()
    plt.plot(false_positive_rate,true_positive_rate,color='darkorange',lw=2,label='ROC curve (AUC=%0.3f)'%roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate ')
    plt.ylabel('True Positive Rate ')
    plt.title('ROC curve for : '+clf['name'])
    plt.legend(loc="lower right")
    plt.savefig('./ClassifierImages/binary/roc/'+clf['name']+'.png')
    plt.show()
    plt.close()

def plot_clf_cmpr(clf_array,labels = ['legit', 'dga']):
    fig = plt.figure(figsize=(25,25))
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    dga_acc_matrix=[]
    for clf_pos,clf in enumerate(clf_array):
        cm  = confusion_matrix(y, clf['y_pred'])
        percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T) 
        dga_acc_matrix.append([max(x) for x in percent])
        for dga_pos,percent_array in enumerate(percent):
            ax.text(dga_pos,clf_pos,max(percent_array),va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))

    cax = ax.matshow(dga_acc_matrix, cmap=plt.cm.Blues)
    pylab.title('Compariosn between models : '+'\n')
    fig.colorbar(cax)
    ax.set_yticklabels([' '] + ['GNB','RFC','KNC','SVM','DTC','XGB','MLP','ADA','QDA'])
    ax.set_xticklabels([' '] + labels)
    pylab.xlabel('Predicted')
    pylab.ylabel('Actual')
#        pylab.savefig('./ClassifierImages/multi/'+clf['name']+'.png')
    print(clf['name'])
    pylab.show()
