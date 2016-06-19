# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:36:52 2016

@author: Wang
"""
import csv
import string
import pandas as pd
from sklearn.cross_validation import  train_test_split
import feature_extract
import sentiment_model

# remove the punctuation
def rm_punctuation(text):
    return text.translate(None, string.punctuation)
    
if __name__ == '__main__':
    # load the data(name, review, rating)
    product = pd.read_csv('amazon_baby.csv',  header='infer')
    #fill all the NA
    product = product.fillna({'review' : ' '}) 
   
    #rm the neural review with rating = 3
    product = product[product.rating != 3]
    
    product = product.dropna( axis = 0, how = 'any')
    
    product['sentiment'] = product['rating'].apply(lambda rating : +1 if rating > 3 else -1)
    #remove the pnctuation
    product['clean_review'] = product['review'].apply(rm_punctuation)
    
    sentiment_data = product['sentiment']
    # split into training sets and test sets
    predict_accuracy = {'logit': [], 'bayes':[], 'svm':[] }
    train_data, test_data = train_test_split(product, test_size = 0.2, random_state = 52, stratify=sentiment_data)
        
    positive_counts = train_data['sentiment'].value_counts()
    print '正面评论所占比例：%f' % (float(positive_counts[1])/positive_counts.sum())   
    #extract feature
    
    print "采用的原始的词频统计特征"
    train_feature_data, test_feature_data = feature_extract.tf(train_data, test_data)    
    #build the model     
    
    accuracy = sentiment_model.sentiment_logit(train_data, test_data, train_feature_data, test_feature_data)
    print "logit模型的预测准确度为:%f" % accuracy
    
    accuracy = sentiment_model.sentiment_bayes(train_data, test_data, train_feature_data, test_feature_data)
    print "bayes模型的预测准确度为:%f" % accuracy
    
    accuracy = sentiment_model.sentiment_svm(train_data, test_data, train_feature_data, test_feature_data)
    print "svm模型的预测准确度为:%f" % accuracy
    
    '''    
    #extract feature
    print "采用的TFIDF"
    train_feature_data, test_feature_data = feature_extract.tfidf(train_data, test_data)    
    #build the model     
    
    accuracy = sentiment_model.sentiment_logit(train_data, test_data, train_feature_data, test_feature_data)
    print "logit模型的预测准确度为:%f" % accuracy
    
    accuracy = sentiment_model.sentiment_bayes(train_data, test_data, train_feature_data, test_feature_data)
    print "bayes模型的预测准确度为:%f" % accuracy
    
    accuracy = sentiment_model.sentiment_svm(train_data, test_data, train_feature_data, test_feature_data)
    print "svm模型的预测准确度为:%f" % accuracy
    '''
        
