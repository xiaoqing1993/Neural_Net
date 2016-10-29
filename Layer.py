# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:02:07 2016

@author: Xiaoqing
"""

import numpy as np
from Node import Node
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer():
    def __init__(self, layer_index, n_node):
        self.layer_index = layer_index
        self.n_node = n_node
        self.node_set = []
        self.outputs = np.zeros(self.n_node)
        self.deltas = np.zeros(self.n_node)
        for node in range(n_node):
            self.node_set.append(Node(self.layer_index, node))
        # self.node_set.append(BiasNode(self.layer_index, n_node))
        
    def get_inputs(self, pre_layer):
        self.x = pre_layer.outputs

    def outputs0(self, original_x):  # for input layer only
        # self.outputs = np.append(original_x, 1)
        self.outputs = original_x
        
    def compute_outputs(self, pre_layer, weights):            
        self.get_inputs(pre_layer)
        for i, node in enumerate(self.node_set):
            node.output_compute(self.x, weights)
            self.outputs[i] = node.oj
    
    def compute_deltas0(self, error): # for output layer only
        for i, node in enumerate(self.node_set):
            node.compute_delta0(error[i])
            self.deltas[i] = node.delta
    
    def compute_deltas(self, weights, next_layer):
        for i, node in enumerate(self.node_set):
            node.compute_delta(weights, next_layer)
            self.deltas[i] = node.delta
        
