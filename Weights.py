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
        self.w = np.random.uniform(-0.1, 0.1, (self.n, self.m))
        self.bias = np.random.uniform(-0.1, 0.1, self.n)
        self.alpha = 1
    def weights_update(self):
        for node in self.down_layer:
            ind = node.node_index
            delta = node.delta
            x = self.down_layer.x
            self.w[ind] = self.w[ind] + self.alpha * delta * x