# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 11:23:39 2017
Testing data loading
@author: Luiza
"""


import json
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
import re
from scipy.sparse import csr_matrix


def clean_str(string):
    
    replacements = {'\n\n':' ', '\n':' '}
    for number, string_number in replacements.items():
        string = string.replace(number, string_number)
    return string    
    
    
def load_pizza_data(folder='../data/pizza', perc_of_train_data=0.8):
    data = json.load(open(folder + '/' + 'train.json', 'r'))
    X = []
    Y = []
    for row in data:
        title_and_text = clean_str(row['request_title'] + ';' + (row['request_text']))
        X.append(title_and_text)
        Y.append(int(row['requester_received_pizza']))
    
    N = len(X)
    true_ratio = true_value_ratio(Y)
    print('Ratio of true values:', true_ratio)
    perm_idx = np.random.permutation(range(N))
    Ntrain = int(perc_of_train_data*N)
    Ntest = N - Ntrain
    Xtrain, Xtest = vectorize_data(X[:Ntrain], X[Ntrain:])
    return np.array(Xtrain.todense(),dtype=np.float32), np.array(Y[:Ntrain],dtype=np.int64), np.array(Xtest.todense(),dtype=np.float32), np.array(Y[Ntrain:],dtype=np.int64)  

def vectorize_data(Xtrain, Xtest):
    ''' Takes as input string data samples
    Returns: a bag of words matrix'''
    
    count_vect = CountVectorizer()
    return count_vect.fit_transform(Xtrain), count_vect.transform(Xtest)

def true_value_ratio(Y):
    true_count = 0
    for value in Y:
        if (value == 1):
            true_count = true_count + 1
    return true_count / len(Y)
    
if __name__ == "__main__":
    Xtrain, Ytrain, Xtest, Ytest = load_pizza_data()
    print(np.shape(Xtrain))
    
    
    
        
    
