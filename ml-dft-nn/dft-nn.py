import numpy as np
from numpy.fft import fft, ifft
np.random.seed(19980430)

D = 785
layer_dims = [D,4096,1024,256,64,10]
L = len(layer_dims) - 1
LEARNING_RATE = 0.01
epochs = 50
minibatch_size = 1000

def relu (Z):
	return np.maximum(0, Z)
def relu_back (dA, Z):
	dZ = np.array(dA, copy=True)
	dZ[Z<=0] = 0
	return dZ
def softmax (Z):
	return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def init_params ():
	params = {}
	for l in range(1,L+1):
		params["W"+str(l)] = np.random.rand(np.max([layer_dims[l-1],layer_dims[l]]),1) * 0.2 - 0.1
	return params

def linear_activation (A_prev, W, next_layer_dim):
	A_prev_pad = np.pad(A_prev, ((0,W.shape[0]-A_prev.shape[0]),(0,0)), 'constant')
	Af_prev = fft(A_prev_pad, axis=0)
	Wf = fft(W, axis=0)
	Zf = np.multiply(Wf, Af_prev)
	Z = np.real(ifft(Zf, axis=0))
	Z = Z[:next_layer_dim,:]
	A = relu(Z)
	cache = (Af_prev, Wf, Z)
	return A, cache

def forw_prop (X, params):
	caches = []
	A = X
	for l in range(1,L+1):
		A, cache = linear_activation(A, params["W"+str(l)], layer_dims[l])
		caches.append(cache)
	A = softmax(A)
	return A, caches

def compute_cost (AL, Y):
	m = Y.shape[1]
	cost = -(1/m) * np.sum(np.multiply(Y, np.log(AL)));
	cost = np.squeeze(cost)
	return cost

def linear_activation_back (dA, cache, prev_layer_dim):
	(Af_prev, Wf, Z) = cache
	dZ = relu_back(dA, Z)
	dZ = np.pad(dZ, ((0,Wf.shape[0]-dZ.shape[0]),(0,0)), 'constant')
	dZf = ifft(dZ, axis=0)
	dWf = np.mean(np.multiply(dZf, Af_prev), axis=1)
	dWf = dWf.reshape((Wf.shape[0],1))
	dW = np.real(fft(dWf, axis=0))
	dAf_prev = np.multiply(dZf, Wf)
	dA_prev_pad = np.real(fft(dAf_prev, axis=0))
	dA_prev = dA_prev_pad[:prev_layer_dim,:]
	return dA_prev, dW

def back_prop (AL, Y, caches):
	grads = {}
	Y = Y.reshape(AL.shape)
	grads["dA"+str(L)] = AL - Y;
	for l in reversed(range(L)):
		grads["dA"+str(l)], grads["dW"+str(l+1)] = linear_activation_back(grads["dA"+str(l+1)], caches[l], layer_dims[l]);
	return grads

def update_params (params, grads):
	for l in range(1,L+1):
		lr = np.std(params["W"+str(l)]) / np.std(grads["dW"+str(l)]) * LEARNING_RATE
		params["W"+str(l)] -= lr * grads["dW"+str(l)]
	return params


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

m_train = mnist.train.images.shape[0]
X_train = np.ones((D,m_train))
X_train[:-1,:] = np.transpose(mnist.train.images)
Y_train = np.transpose(mnist.train.labels)
m_test = mnist.test.images.shape[0]
X_test = np.ones((D,m_test))
X_test[:-1,:] = np.transpose(mnist.test.images)
Y_test = np.transpose(mnist.test.labels)

params = init_params()
_acc_train = []
_acc_test = []
_cost = []
for e in range(epochs):
	acc_train_ = []
	for b in range(m_train // minibatch_size):
		X_train_mb = X_train[:, (b*minibatch_size):((b+1)*minibatch_size)]
		Y_train_mb = Y_train[:, (b*minibatch_size):((b+1)*minibatch_size)]
		AL, caches = forw_prop(X_train_mb, params)
		cost = compute_cost(AL, Y_train_mb)
		_cost.append(cost)
		grads = back_prop(AL, Y_train_mb, caches)
		params = update_params(params, grads)
		acc_train = np.mean(np.argmax(Y_train_mb,0) == np.argmax(AL,0))
		acc_train_.append(acc_train)
		print("Epoch %d\tMinibatch %d\tLoss %1.4f\tAcc_train %1.4f" %(e, b, cost, acc_train))
	AL, caches = forw_prop(X_test, params)
	acc_test = np.mean(np.argmax(Y_test,0) == np.argmax(AL,0))
	_acc_train.append(np.mean(acc_train_))
	_acc_test.append(acc_test)
	print("Acc_train %1.4f\tAcc_test %1.4f" %(np.mean(acc_train_), acc_test))
	LEARNING_RATE *= 0.95
print(_acc_train)
print(_acc_test)
print(_cost)

