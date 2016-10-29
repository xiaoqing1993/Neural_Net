# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:15:37 2016

@author: Xiaoqing
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''
x = np.array(range(-100, 100)) / 10.0
y = sigmoid(x)
'''
class Node():
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.oj = 0

    def output_compute(self, x, weights):
        w_up = weights.w[self.node_index]
        y = np.dot(x, w_up) + weights.bias[self.node_index]
        self.oj = sigmoid(y)
 
    def compute_delta0(self, error): # for output nodes only
        self.delta = self.oj * (1 - self.oj) * error
        
    def compute_delta(self, weights, next_layer): 
        w_down = weights.w[:, self.node_index]
        self.delta = self.oj * (1 - self.oj) * np.dot(w_down, next_layer.deltas) 