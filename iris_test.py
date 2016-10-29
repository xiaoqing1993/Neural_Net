# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:43:18 2016

@author: Xiaoqing
"""
from NeuralNet import NeuralNet
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target


combo = zip(X,y)
np.random.shuffle(combo)
X, y = zip(*combo)
X = np.array(X)
y = np.array(y)


y_new = []
n_label = len(set(y))
for label in y:
    y_temp = np.zeros(n_label)
    y_temp[label] = 1
    y_new.append(y_temp)

y_new = np.array(y_new)

neural_network = NeuralNet(n_Hlayer = 2, n_node = 100, alpha = 1)
neural_network.train(X, y_new, 10)
y_p = neural_network.predict(X)
y_p = np.array(y_p)
acc = accuracy_score(y, y_p)
