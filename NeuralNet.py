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
    def __init__(self, n_Hlayer, n_node, alpha, activation):
        self.n_Hlayer = n_Hlayer
        self.n_node = n_node
        n_node_in = 784
        self.n_node_out = 10
        self.val_acc = 0
        self.alpha = alpha
        self.activation = activation
        self.Net = [Layer(0, n_node_in, self.activation)]
        self.WR = []
        for layer in range(1, n_Hlayer+1):
            self.Net.append(Layer(layer, self.n_node, self.activation))
        if self.activation == 'sigmoid' : 
            self.Net.append(Layer(n_Hlayer+1, self.n_node_out, self.activation))
        elif self.activation == 'relu' :
            self.Net.append(Layer(n_Hlayer+1, self.n_node_out, 'softmax'))
        for layer_ind in range(n_Hlayer+1):
            self.WR.append(Weights(self.Net[layer_ind], self.Net[layer_ind+1]))
    
    def ytrain_preprocess(self, y):
        y_new = []
        n_label = self.n_node_out
        for label in y:
            y_temp = np.zeros(n_label)
            y_temp[label] = 1
            y_new.append(y_temp)
        y_new = np.array(y_new)
        return y_new
   
    def FeedForward(self, sample):
        self.Net[0].outputs0(sample)
        for index in range(1, self.n_Hlayer+2):
            self.Net[index].compute_outputs(self.Net[index-1], self.WR[index-1])
        self.y_pre = self.Net[-1].outputs
        yp = self.Net[-1].outputs
        return yp
        
    def BackPropagation(self, y):
        #output_layer = self.Net[-1]
        yp = self.Net[-1].outputs
        error = y - yp
        self.Net[-1].compute_deltas0(error)
        for layer in reversed(self.Net[:-1]):
            ind = layer.layer_index
            #next_layer = self.Net[ind+1]
            self.Net[ind].compute_deltas(self.WR[ind], self.Net[ind+1])
    
    def WeightUpdate(self):
        for ind in range(self.n_Hlayer+1):
            self.WR[ind].weights_update(self.alpha)
            
    def train_a_sample(self, sample, y):
        yp = self.FeedForward(sample)
        self.BackPropagation(y)
        self.WeightUpdate()
            
    def train(self, X, y, iteration):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
        y_new = self.ytrain_preprocess(y_train)
        self.Xtest = X_test
        self.ytest = y_test
        self.tra_acc = np.zeros(iteration)
        self.val_acc = np.zeros(iteration)
        for index in range(iteration):
            print("iteration: ", index)
            for s in range(len(y_train)):
                self.train_a_sample(X_train[s], y_new[s])
            yp_train = self.predict(X_train)
            self.tra_acc[index] = accuracy_score(y_train, yp_train)
            print('training accuracy:', self.tra_acc[index])
            yp_test = self.predict(X_test)
            self.val_acc[index] = accuracy_score(y_test, yp_test)
            print('validation accuracy:', self.val_acc[index])
            
    def predict(self, X):
        m, n = np.shape(X)
        y_p = np.zeros(m)
        for ind, sample in enumerate(X):
            #yp = copy.deepcopy(self.FeedForward(sample))
            yp = self.FeedForward(sample)
            y_p[ind] = np.argmax(yp)
        return y_p

