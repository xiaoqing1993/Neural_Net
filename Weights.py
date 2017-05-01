# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:44:01 2016

@author: Xiaoqing
"""

import numpy as np

class Weights():
    def __init__(self, up_layer, down_layer):
        self.up_layer = up_layer
        self.down_layer = down_layer
        self.m = up_layer.n_node
        self.n = down_layer.n_node
        self.w = np.random.normal(0, 0.1, (self.n, self.m))
        self.bias = np.ones(self.n) * 0.1
        self.x = np.zeros(self.m)
        #self.gradients = np.zeros((self.n, self.m+1))
    def weights_update(self, alpha):
        self.x = self.up_layer.outputs
        for node in self.down_layer.node_set:
            ind = node.node_index
            delta = node.delta
            gradients = delta * self.x 
            self.w[ind] -=  alpha * gradients
            self.bias[ind] -= alpha * delta
            #self.gradients[ind] = np.append(gradients, -delta)
