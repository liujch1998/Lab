import numpy as np
np.set_printoptions(threshold=np.inf)

L = 3
layer_dims = [785,100,30,10]
lr = 0.1
epochs = 1000

def relu (Z):
	return np.maximum(0,Z)
def relu_back (dA, Z):
	dZ = np.array(dA, copy=True)
	dZ[Z<=0] = 0
	return dZ
def softmax (Z):
	return np.exp(Z) / np.sum(np.exp(Z),axis=0)

def init_params ():
	params = {}
	for l in range(1,L+1):
		params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
	return params

def linear_activation (A_prev, W):
	Z = np.dot(W, A_prev)
	A = relu(Z)
	cache = ((A_prev, W), Z)
	return A, cache

def forw_prop (X, params):
	caches = []
	A = X
	for l in range(1,L+1):
		A, cache = linear_activation(A, params['W'+str(l)])
		caches.append(cache)
	A = softmax(A)
	return A, caches

def compute_cost (AL, Y):
	m = Y.shape[1]
	cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)));
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	return cost

def linear_activation_back (dA, cache):
	((A_prev, W), Z) = cache
	dZ = relu_back(dA, Z)
	m = A_prev.shape[1]
	dW = np.dot(dZ, A_prev.T) / m
	dA_prev = np.dot(W.T, dZ)
	return dA_prev, dW, dZ

def back_prop (AL, Y, caches):
	grads = {}
	Y = Y.reshape(AL.shape)
	grads["dA"+str(L)] = AL - Y;
	for l in reversed(range(L)):
		grads["dA"+str(l)], grads["dW"+str(l+1)], grads["dZ"+str(l+1)] = linear_activation_back(grads["dA"+str(l+1)], caches[l]);
	return grads

def update_params (params, grads, lr):
	for l in range(1,L+1):
		params["W"+str(l)] -= lr * grads["dW"+str(l)]
	return params


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_train = np.ones((785,55000))
X_train[:-1,:] = np.transpose(mnist.train.images)
Y_train = np.transpose(mnist.train.labels)
X_test = np.ones((785,10000))
X_test[:-1,:] = np.transpose(mnist.test.images)
Y_test = np.transpose(mnist.test.labels)

params = init_params()
for e in range(epochs):
	AL, caches = forw_prop(X_train, params)
	cost = compute_cost(AL, Y_train)
	grads = back_prop(AL, Y_train, caches)
	if e < 50:
		params = update_params(params, grads, lr*10)
	else:
		params = update_params(params, grads, lr)
	acc_train = np.sum(np.argmax(Y_train,0) == np.argmax(AL,0)) / X_train.shape[1]
	
	AL, caches = forw_prop(X_test, params)
	acc_test = np.sum(np.argmax(Y_test,0) == np.argmax(AL,0)) / X_test.shape[1]

	print("Epoch %d\tLoss %1.4f\tAcc_train %1.4f\tAcc_test %1.4f" %(e, cost, acc_train, acc_test))

