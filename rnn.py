# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:43:05 2017

@author: AnhNQ17
"""
import numpy as np


#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#%%

"""
Define number of node in 3 hidden layers
"""
#L = [4, 5, 3, 5, 4]
L = [3, 5, 3, 5, 3]
#L = [2, 5, 3, 5, 2]
#%%
def initMat(L):
    np.random.seed(0)
    Mat = []
    for i in range(0, len(L)-1):
        Mat.append(np.matrix(np.random.random((L[i], L[i+1]))))
    return Mat   
#%%   
Mat = initMat(L)
Mat[0] = Mat[0]/5
Mat[1] = Mat[1]/5
#%%
def flatListMatrix(Mat):
    c = np.matrix([]).getT()
    for i in range(0, len(Mat)):
        m = Mat[i].flatten().getT()
        c = np.concatenate((c, m))
    return c
#%%
def deflatListMatrix(c, L):
    Mat = []
    last_index = 0
    for i in range(0, len(L)-1):
        m = c[last_index : last_index + (L[i]*L[i+1])].reshape((L[i], L[i+1]))
        Mat.append(m)
        last_index = L[i]*L[i+1]
    return Mat
#%%
a = [1, 10, 1]
N = 4
def forwardActivationFunction(i, x):
   if (i == 1 or i == 3):
       return np.tanh(a[i-1] * x)
   elif i == 2:
       return  (sum(np.tanh(a[i-1] * (x - j / N)) for j in range(1, N)) + (N-1))/(2 * (N-1))
       #return sigmoid(a[i-1]*x)
   elif i == 0:
       return x
   else: #seperate this for experiment prupose. The last one could be sigmoid or Id 
       #return x
       return sigmoid(x)

#%%

def backwardPropagationDerivativeFunction(i, x):
    if (i==0 or i==2):
        t = 1 - np.tanh(a[i] * x)
        return a[i] * (1 - np.multiply(t, t))
    elif i == 1:
        return a[i]/(2*(N - 1)) * sum((1- np.square(np.tanh(a[i] * (x - j / N)))) for j in range(1, N))
        #return a[i]/(2*(N - 1)) * sum((1- np.multiply(1 - np.tanh(a[i] * (x - j / N)), 1 - np.tanh(a[i] * (x - j / N)))) for j in range(1, N))
        #return a[i]*np.multiply(sigmoid(a[i]*x), (1 - sigmoid(a[i] * x)))
#%%

def forwardPropagation(Mat, data):
    if type(data) is not np.ndarray:
        r = np.array(data)
    else:
        r = data
    z = []
    for i in range(0, len(Mat)):
        multiplication = np.dot(r, Mat[i])
        z.append(multiplication)
        r = forwardActivationFunction(i, multiplication)
#        print(r)
    return r, outlierFactor(data, r)

#%%
def outlierFactor(din, dout):
    n = len(din)
    diff = din - dout
    return 1/n * np.multiply(diff, diff).sum()

#%%
def cost(Mat, data):
    if type(data) is not np.ndarray:
        r = np.array(data)
    else:
        r = data
    z = []
    for i in range(0, len(Mat)):
        multiplication = np.dot(r, Mat[i])
        z.append(multiplication)
        r = forwardActivationFunction(i, multiplication)
        
    vcost = np.matrix(data).flatten() - r.flatten()
    vcost = np.multiply(vcost, vcost).sum() / (r.shape[0]*r.shape[1])
    print("Cost:", vcost)
#==============================================================================
#     print("R:", r)
#     print("data:", np.matrix(data).flatten())
#==============================================================================
    delta = r - np.matrix(data)
    delta = delta.getT()
    grad = (delta * z[-2]).getT()
    theta_grad = grad.flatten().getT()
#==============================================================================
#     print("THETA:", theta_grad)
#     print("Size:", grad.shape)
#==============================================================================
    for i in range(len(Mat)-1, 0, -1):
#==============================================================================
#         print("i:",i)
#         print("grad:", grad)
#         print("Mat:", Mat[i])
#         print("Z:", z[i-1])
#         print("product:",Mat[i] * grad)
#==============================================================================
#        print("Pre Delta:", delta)
        delta = np.multiply(Mat[i] * delta, backwardPropagationDerivativeFunction(i-1, z[i-1].getT()))
#        print("Delta:", delta)
#        print("Mat:", Mat[i])
#        print("Z:", z[i-1].getT())
#        print("I:", i)
#        print("f Z:", backwardPropagationDerivativeFunction(i-1, z[i-1].getT()))
        if i>=2:
            grad = (delta * forwardActivationFunction(i-2, z[i-2])).getT()
        else:
            grad = (delta * data).getT()
        grad = grad/len(data)
#==============================================================================
#         print("Size:", grad.shape)
#         print("THETA:", grad.flatten())
#==============================================================================
        theta_grad = np.concatenate((grad.flatten().getT(), theta_grad))
    return vcost, theta_grad
#%%
 l_rate = 0.0001
 l_max = 0.02
 l_reduction_factor = 0.98
 l_enlargement_factor = 1.005
 
#%% 
def trainModel(Mat, data, steps=100):
    precost = -1
    prerate = -1
    M = Mat
    for i in range(0, steps):
        vcost, theta_grad = cost(M, data)
        learning_rate = 0
        if prerate < 0:
            learning_rate = l_rate
        elif vcost > 1.01 * precost:
            learning_rate = l_reduction_factor * prerate
        elif (vcost < precost and learning_rate < l_max):
            learning_rate = l_enlargement_factor * prerate
        else:
            learning_rate = prerate
            
        theta = flatListMatrix(M) - learning_rate * theta_grad
        print("learning rate:", learning_rate)
#        print("gradient:", theta_grad)
        M = deflatListMatrix(theta, L)
        precost = vcost
        prerate = learning_rate
    return M
    
#%%
def calculateError(Mat, data):
    r = []
    for i in range(0, len(data)):
        output, error = forwardPropagation(Mat, data[i])
        r.append(error)
    return r
    

#%%
M = trainModel(Mat, [[0.1, 0.3], [0.1, 0.4], [0.1, 0.3], [0.1, 0.8], [0.1, 0.8], [0.1, 0.8], [0.1, 0.8], [0.1, 0.8], [0.1, 0.8], [0.1, 0.8]], steps=100)


#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
#%%
factor = 20
dftempt = pd.read_csv('file_case1.csv')
dftempt.drop('sensor1', axis=1, inplace=True)
dftempt = dftempt/factor;
data_array = dftempt.as_matrix()
#%%
dftest = pd.read_csv('file_case2.csv').drop(['sensor1', 'sensor2'], axis=1) 
#dftest = dftest/factor;
data_test = dftest.as_matrix()
#%%
dftest2 = pd.read_csv('file_case3.csv').drop(['sensor1', 'sensor2', 'sensor4'], axis=1) 
#dftest2 = dftest2/factor;
data_test2 = dftest2.as_matrix()
#%%
def test(f, x):
    return f(x)

def f(x):
    return x**2
#%%
































