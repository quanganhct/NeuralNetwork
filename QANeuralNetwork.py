# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:57:05 2017

@author: AnhNQ17
"""

import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        
    def addLayer(self, _layer):
        self.layers.append(_layer)
        
    def getSizeLayers(self, X, Y):
        _size = []
        _size.append(len(X[0]))
        for i in range(0, len(self.layers)):
            _size.append(self.layers[i].size)
        _size.append(len(Y[0]))
        return _size
        
    def initMat(self, L):
        np.random.seed(0)
        Mat = []
        for i in range(0, len(L)-1):
            Mat.append(np.random.random((L[i], L[i+1])))
        return Mat 
        
    def flatListMatrix(self, Mat):
        c = np.array([])
        for i in range(0, len(Mat)):
            m = Mat[i].flatten()
            c = np.concatenate((c, m))
        return c
        
    def deflatListMatrix(self, c, L):
        Mat = []
        last_index = 0
        for i in range(0, len(L)-1):
            m = c[last_index : last_index + (L[i]*L[i+1])].reshape((L[i], L[i+1]))
            Mat.append(m)
            last_index = L[i]*L[i+1]
        return Mat
        
    def cost(self, Mat, data):
        if type(data) is not np.ndarray:
            r = np.array(data)
        else:
            r = data
        z = []
        a = []
        for i in range(0, len(Mat)):
            multiplication = np.dot(r, Mat[i])
            z.append(multiplication)
            r = self.layers[i].forward(multiplication)
            a.append(r)
            
        vcost = np.array(data).flatten() - r.flatten()
        vcost = (vcost**2).sum() / (r.shape[0]*r.shape[1])
        print("Cost:", vcost)
        delta = r - np.matrix(data)
        delta = delta.getT()
        grad = (delta * z[-2]).getT()
        theta_grad = grad.flatten().getT()
        for i in range(len(Mat)-1, 0, -1):
            delta = np.dot(Mat[i], delta) * backwardPropagationDerivativeFunction(i-1, z[i-1].getT()))
            if i>=2:
                grad = (delta * forwardActivationFunction(i-2, z[i-2])).getT()
            else:
                grad = (delta * data).getT()
            grad = grad/len(data)
            theta_grad = np.concatenate((grad.flatten().getT(), theta_grad))
        return vcost, theta_grad
        
    def fit(self, X, Y, iterations=300):
        if ((type(X) is not list) or (len(np.array(X).shape) != 2)):
            raise ValueError('X must be 2D list')
        if ((type(Y) is not list) or (len(np.array(Y).shape) != 2)):
            raise ValueError('Y must be 2D list')
            
        S = self.getSizeLayers(X, Y)
        Theta = self.initMat(S)
        
        
    def forward(self, data):
        return 0
       
#%%
class HiddenLayer:
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))
        
    def sigmoidDerivation(X):
        Y = HiddenLayer.sigmoid(X)
        return Y * (1 - Y)
    
    def __init__(self, size, fs=sigmoid, dfs=sigmoidDerivation):
        self.size = size
        if type(fs) is list:
            self.fs = fs
            self.dfs = dfs
        else:
            self.fs = [fs]*size
            self.dfs = [fs]*size

    def forward(self, X):
        _input = X
        if type(X) is not np.ndarray:
            _input = np.array(X)
        _input = _input.T
        if self.size != _input.shape[0]:
            raise ValueError('Wrong input size')
        l = []
        for i in range(0, self.size):
            m = self.fs[i](_input[i])
            l.append(m)
        return np.array(l).T

    def backward(self, X):
        _input = X
        if type(X) is not np.ndarray:
            _input = np.array(X)
        _input = _input.T
        if self.size != _input.shape[0]:
            raise ValueError('Wrong input size')
        l = []
        for i in range(0, self.size):
            m = self.dfs[i](_input[i])
            l.append(m)
        return np.array(l).T

#%%
    