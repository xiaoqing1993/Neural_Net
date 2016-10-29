# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:36:17 2016

@author: Xiaoqing
"""
import Layer
import Weights
class NeuralNet():
    def __init__(self, n_Hlayer, n_node):
        self.n_Hlayer = n_Hlayer
        self.n_node = n_node
        n_node_in = 3600
        n_node_out = 19
        self.Net = [Layer(0, n_node_in)]
        for layer in range(1, n_Hlayer+1):
            self.Net.append(Layer(layer, self.n_node))
        self.Net.append(Layer(n_Hlayer+1, n_node_out))
    
    def FeedForward(self, sample):
        self.Net[0].outputs0(sample)
        for ind, layer in enumerate(self.Net[1:], 1):
            pre_layer = self.Net[ind-1]
            WR = Weights(pre_layer, layer)
            layer.compute_outputs(pre_layer, WR)
        self.y_pre = self.Net[-1].outputs
 
    def BackPropagation(self, sample, y):
        output_layer = self.Net[-1]
        error = y - output_layer.ouputs
        output_layer.compute_deltas0(error)
        for layer in reversed(self.Net[:-1]):
            ind = layer.layer_index
            next_layer = self.Net[ind+1]
            WR = Weights(layer, next_layer)
            layer.compute_deltas(WR, next_layer)
   
'''
to do list:
    - implement weights update
    - add bias term update
    - train and test
'''         