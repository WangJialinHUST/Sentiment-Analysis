# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:44:27 2016

@author: Wang
"""
import csv
import string
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
#the result
#def data_preprocessing():
 # load the data(name, review, rating)

'''
product = pd.read_csv('amazon_baby.csv',  header='infer')
print  '总共有%d 条评论 \n' % len(product)

#fill all the NA
product = product.fillna({'review' : ' '}) 
   
#rm the neural review with rating = 3
product = product[product.rating != 3]

print "有效评论为 %d 条 \n" % len(product)
product = product.dropna( axis = 0, how = 'any')

product['sentiment'] = product['rating'].apply(lambda rating : +1 if rating > 3 else -1)

positive_counts = product['sentiment'].value_counts()

print '正面评论所占比例：%f' % (float(positive_counts[1])/positive_counts.sum())   
    
'''     

def result_show():
    plt.figure
    tf = [0.841031, 0.930554, 0.907575, 0.920551]
    x = [1,2,3,4]
    plt.bar(x, tf,width=0.8,align='center')
  #  plt.plot(x, a, 'ro-', x, b, 'bs-', x,c , 'gd-', linewidth=2.0, markersize = 8.0)
    plt.ylabel('Prediction Accuracy')
    plt.ylim([0.8, 1])
    plt.xlabel('different algorithm')
    plt.xticks((1, 2,3,4),('baseline','logit regression', 'naive bayes', 'svm'))
    plt.show()
    
    plt.figure
    tf = [0.930554, 0.907575, 0.920551]
    tfidf = [0.931485, 0.841493, 0.932927]
    width = 0.4
    x = [1,2,3]
    plt.bar(x, tf,width,color = 'g', label = 'TF')
    plt.bar([i + width for i in x ], tfidf,width, color = 'b', label = 'TFIDF')
  #  plt.plot(x, a, 'ro-', x, b, 'bs-', x,c , 'gd-', linewidth=2.0, markersize = 8.0)
    plt.ylabel('Prediction Accuracy')
    plt.ylim([0.8, 1])
    plt.xlabel('different algorithm')
    plt.legend(['TF', 'TFIDF'], loc='upper right')
    plt.xticks((1.4, 2.4,3.4),('logit regression', 'naive bayes', 'svm'))
    plt.show()
    
    plt.figure
    unigram = [0.930554, 0.907575, 0.920551]
    bigram = [0.947375, 0.902139, 0.943049]
    width = 0.4
    x = [1,2,3]
    plt.bar(x, unigram,width,color = 'g', label = 'TF')
    plt.bar([i + width for i in x ], bigram,width, color = 'b', label = 'TFIDF')
  #  plt.plot(x, a, 'ro-', x, b, 'bs-', x,c , 'gd-', linewidth=2.0, markersize = 8.0)
    plt.ylabel('Prediction Accuracy')
    plt.ylim([0.8, 1])
    plt.xlabel('different algorithm')
    plt.legend(['Unigram', 'Bigram'], loc='upper right')
    plt.xticks((1.4, 2.4,3.4),('logit regression', 'naive bayes', 'svm'))
    plt.show()
    
    plt.figure
    no_balance = [0.930554,  0.920551]
    balance = [0.915986, 0.908386]
    width = 0.4
    x = [1,2]
    plt.bar(x, no_balance,width,color = 'g', label = 'TF')
    plt.bar([i + width for i in x ], balance,width, color = 'b', label = 'TFIDF')
  #  plt.plot(x, a, 'ro-', x, b, 'bs-', x,c , 'gd-', linewidth=2.0, markersize = 8.0)
    plt.ylabel('Prediction Accuracy')
    plt.ylim([0.8, 1])
    plt.xlabel('different algorithm')
    plt.legend(['No Class Balance', 'Class Balance'], loc='upper right')
    plt.xticks((1.4, 2.4 ),('logit regression',  'svm'))
    plt.show()
    

result_show()









