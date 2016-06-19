# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 23:27:33 2016

@author: Wang
"""
from sklearn.feature_extraction import text

def tfidf(train_data, test_data):
    vectorizer = text.TfidfVectorizer(token_pattern = r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(train_data['clean_review'])
    test_matrix = vectorizer.transform(test_data['clean_review'])
    return train_matrix, test_matrix
#
def tf(train_data, test_data):
    vectorizer = text.CountVectorizer(token_pattern = r'\b\w+\b',ngram_range=(1,2))
    train_matrix = vectorizer.fit_transform(train_data['clean_review'])
    test_matrix = vectorizer.transform(test_data['clean_review'])
    return train_matrix, test_matrix
