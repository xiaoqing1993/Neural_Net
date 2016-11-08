# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 19:54:34 2016

@author: Xiaoqing
"""
from NeuralNet import NeuralNet
from read_data import ReadData
import numpy as np

r = ReadData()
X, y = r.read_train()

print 'time', 0
n_layer0 = np.random.randint(3, 8)
n_node_perlayer0 = np.random.randint(20, 300)
learning_rate0 = np.random.uniform(0, 1.5)
neural_network = NeuralNet(n_Hlayer = n_layer0-2, n_node = n_node_perlayer0, alpha = learning_rate0)
print 'number of hidden layer: ', neural_network.n_Hlayer
print 'number of nodes per hidden layer: ', neural_network.n_node
print 'learning rate: ', neural_network.alpha
print 'training...'
neural_network.train(X, y, 5)
acc0 = neural_network.val_acc
print '\n'

for i in range(9):
    print 'time', i+1
    if n_layer0-2 > 2:
        n_layer = np.random.randint(n_layer0-2, n_layer0+2)
    else:
        n_layer = np.random.randint(3, n_layer0+2)
    
    if n_node_perlayer0 > 20 :
        n_node_perlayer = np.random.randint(n_node_perlayer0-20, n_node_perlayer0+20)
    else:
        n_node_perlayer = np.random.randint(20, n_node_perlayer0+20)
    if learning_rate0 > 0.2:
        learning_rate = np.random.uniform(learning_rate0-0.2, learning_rate0+0.2)
    else:
        learning_rate = np.random.uniform(0, learning_rate0+0.2)
    neural_network = NeuralNet(n_Hlayer = n_layer-2, n_node = n_node_perlayer, alpha = learning_rate)
    print 'number of hidden layer: ', neural_network.n_Hlayer
    print 'number of nodes per hidden layer: ', neural_network.n_node
    print 'learning rate: ', neural_network.alpha
    print 'training...'
    neural_network.train(X, y, 5)
    acc = neural_network.val_acc
    print '\n'
    if acc > acc0:
        acc0 = acc
        n_layer0 = n_layer
        n_node_perlayer0 = n_node_perlayer
        learning_rate0 = learning_rate

'''
print 'calculating training accuracy...'
y_p = neural_network.predict(X_train)
y_p = np.array(y_p)
acc_train = accuracy_score(y_train, y_p)
print 'training accuracy = ', acc_train

print 'validating...'
yp_test = neural_network.predict(X_test)
yp_test = np.array(yp_test)
acc_test = accuracy_score(y_test, yp_test)
print 'validation accuracy = ', acc_test
'''

