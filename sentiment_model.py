# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 23:26:31 2016

@author: Wang
"""
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#logistic regression model
def sentiment_logit(train_data, test_data, train_matrix, test_matrix):
    #build the model class_weight= {+1: 0.841033 , -1:0.15897}
    sentiment_model = linear_model.LogisticRegression()
    sentiment_model.fit(train_matrix,  train_data['sentiment'])
    #evaluate the model
    return sentiment_model.score(test_matrix, test_data['sentiment'])


#naive bayes classification model 
def sentiment_bayes(train_data, test_data, train_matrix, test_matrix):
    sentiment_model = MultinomialNB()
    sentiment_model.fit(train_matrix, train_data['sentiment'])    
    return sentiment_model.score(test_matrix, test_data['sentiment'])
        
    
#svm modelclass_weight={+1:1.189, -1:6.291}
def sentiment_svm(train_data, test_data, train_matrix, test_matrix):
    sentiment_model = svm.LinearSVC()
    sentiment_model.fit(train_matrix, train_data['sentiment'])    
    return sentiment_model.score(test_matrix, test_data['sentiment'])