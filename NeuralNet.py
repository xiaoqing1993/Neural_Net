# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:36:17 2016

@author: Xiaoqing
"""
from Layer import Layer
from Weights import Weights
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import copy
class NeuralNet():
    def __init__(self, n_Hlayer, n_node, alpha):
        self.n_Hlayer = n_Hlayer
        self.n_node = n_node
        n_node_in = 3600
        n_node_out = 19
        self.alpha = alpha
        self.Net = [Layer(0, n_node_in)]
        self.WR = []
        for layer in range(1, n_Hlayer+1):
            self.Net.append(Layer(layer, self.n_node))
        self.Net.append(Layer(n_Hlayer+1, n_node_out))
        
        for layer_ind in range(n_Hlayer+1):
            self.WR.append(Weights(self.Net[layer_ind], self.Net[layer_ind+1]))
    
    def ytrain_preprocess(self, y):
        y_new = []
        n_label = len(set(y))
        for label in y:
            y_temp = np.zeros(n_label)
            y_temp[label] = 1
            y_new.append(y_temp)
        y_new = np.array(y_new)
        return y_new
   
    def FeedForward(self, sample):
        self.Net[0].outputs0(sample)
        for ind, layer in enumerate(self.Net[1:], 1):
            pre_layer = self.Net[ind-1]
            layer.compute_outputs(pre_layer, self.WR[ind-1])
        self.y_pre = self.Net[-1].outputs
        yp = self.Net[-1].outputs
        return yp
        
    def BackPropagation(self, y):
        output_layer = self.Net[-1]
        error = y - self.y_pre
        #print 'error = ', error
        output_layer.compute_deltas0(error)
        for layer in reversed(self.Net[:-1]):
            ind = layer.layer_index
            next_layer = self.Net[ind+1]
            layer.compute_deltas(self.WR[ind], next_layer)
    
    def WeightUpdate(self):
        for weight in self.WR:
            weight.weights_update(self.alpha)
            #print weight.w
            
    def train_epoch(self, X, y):
        for ind, sample in enumerate(X):
            sample = sample.toarray()
            #print ind, sample
            yp = self.FeedForward(sample)
            self.BackPropagation(y[ind])
            self.WeightUpdate()
            
    def train(self, X, y, iteration):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        y_new = self.ytrain_preprocess(y_train)
        for index in range(iteration):
            print 'epoch', index
            self.train_epoch(X_train, y_new)
            self.iteration_predict(X_train, y_train)
            yp_test = self.predict(X_test)
            print 'validation error:', accuracy_score(y_test, yp_test)
            
    def predict(self, X):
        m, n = np.shape(X)
        y_p = np.zeros(m)
        #y_p = []
        for ind, sample in enumerate(X):
            sample = sample.toarray()
            #print ind, sample
            yp = copy.deepcopy(self.FeedForward(sample))
            #print self.y_pre
            #print yp
            #y_p.append(yp)
            y_p[ind] = np.argmax(yp)
        return y_p
        
    def iteration_predict(self, X, yo):
        m, n = np.shape(X)
        y_p = np.zeros(m)
        for ind, sample in enumerate(X):
            sample = sample.toarray()
            yp = copy.deepcopy(self.FeedForward(sample))
            y_p[ind] = np.argmax(yp)
        print 'training accuracy:', accuracy_score(yo, y_p)

'''
to do list
    - def predict / self.y_pre ???
'''