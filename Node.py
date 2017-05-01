# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:15:37 2016

@author: Xiaoqing
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0.0, x)

def delta_sigmoid(oj, w_down, deltas_down):
    return oj * (1 - oj) * np.dot(w_down, deltas_down) 

def delta_relu(oj, w_down, deltas_down):
    if oj != 0 :
        return np.dot(w_down, deltas_down)
    else :
        return 0

def delta_cross_entropy(error): # error = target_i - ypredict_i
    return -error

'''
x = np.array(range(-100, 100)) / 10.0
y = sigmoid(x)
'''
class Node():
    def __init__(self, layer_index, node_index, activation):
        self.layer_index = layer_index
        self.node_index = node_index
        self.oj = 0
        self.delta = 0
        self.activation = activation

    def output_compute(self, x, weights):
        w_up = weights.w[self.node_index]
        y = np.dot(x, w_up) + weights.bias[self.node_index]
        if self.activation == 'sigmoid' :
            self.oj = sigmoid(y)
        elif self.activation == 'relu' :
            self.oj = relu(y)
        elif self.activation == 'softmax' : # oj/sum in layer level
            self.oj = np.exp(y)  # exp modification happens
        else :
            raise(TypeError('Please specify an activation function type (sigmoid or relu)'))
    
    def compute_delta0(self, error): # for output nodes only # error = target_i - ypredict_i
        self.delta = delta_cross_entropy(error)  # = yk - tk

        
        #self.delta = self.oj * (1 - self.oj) * error
        
    def compute_delta(self, weights, next_layer): 
        w_down = weights.w[:, self.node_index]
        if self.activation == 'sigmoid' :
            self.delta = delta_sigmoid(self.oj, w_down, next_layer.deltas)
        elif self.activation == 'relu' :
            self.delta = delta_relu(self.oj, w_down, next_layer.deltas)
        else :
            raise(TypeError('Please specify an activation function type (sigmoid or relu)'))

        #self.delta = self.oj * (1 - self.oj) * np.dot(w_down, next_layer.deltas) 