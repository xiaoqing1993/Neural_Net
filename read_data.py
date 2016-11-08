# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:52:46 2016

@author: Xiaoqing
"""

import numpy as np
import csv
from scipy import sparse
class ReadData():
    def __init__(self, width = 60, train_size = 100000, test_size = 20000):
        self.width = width
        self.train_size = train_size
        self.test_size = test_size
    
    def binarize(self, X):
        s = np.shape(X)
        Xb = np.uint8(np.zeros(s))
        Xb[X==255] = 1
        return Xb
        
    def read_train(self):
        print 'reading x ...'
        x = np.fromfile('train_x.bin', dtype='uint8')
        x = x.reshape((self.train_size,self.width,self.width))
        Xnew = x.reshape(self.train_size, self.width ** 2)
        Xb = self.binarize(Xnew)
        Xs = sparse.csr_matrix(Xb)
        print 'reading y ...'
        filename = 'train_y.csv'
        with open(filename,'rb') as originaldata:
            categ_info = csv.reader(originaldata)
            y = list(categ_info)
        y.pop(0)
        for index, row in enumerate(y):
            y[index] = int(row[1])
        y = np.array(y)
        return Xs, y
        
    def read_test(self):
        print 'reading Kaggle test data ...'
        x = np.fromfile('test_x.bin', dtype='uint8')
        x = x.reshape((self.test_size, self.width, self.width))
        Xnew = x.reshape(self.test_size, self.width ** 2)
        Xb = self.binarize(Xnew)
        Xs = sparse.csr_matrix(Xb)
        return Xs
#plt.imshow(x[-1])
#plt.show()


