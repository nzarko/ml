#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:00:04 2018

@author: nick
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

def zarko_scores(y_test, y_pred):
    precision = precision_score(y_test, y_pred)*100
    recall = recall_score(y_test, y_pred)*100
    f1score = f1_score(y_test, y_pred)*100
    acc = accuracy_score(y_test, y_pred)*100

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    print('Model',end='\t\t')
    for m in metrics:
        print(m,end='\t')
        
    print()
    print(model_name, end='\t')
    print('{:<6.2f}'.format(acc),end='\t\t')
    print('{:<6.2f}'.format(precision),end='\t\t')
    print('{:<6.2f}'.format(recall), end='\t\t')
    print('{:<6.2f}'.format(f1score))
    print('End of results')
        
    
def setTitle(classifier):
    title = '\t' + '*'*4 + ' NLP {} Report '.format(classifier) + '*'*4
    print(title)
    
model_name = ''
def set_model_name(name):
    global model_name
    model_name = name