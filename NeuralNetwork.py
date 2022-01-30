"""
	A Multilayer Fully connected deep Neural Network model.
	Author: Aditya Kahol
"""
import numpy as np
from random import choice

def sigmoid(A):
	r,c = A.shape
	for i in range(r):
		for j in range(c):
			A[i][j] = np.round(1/(1+np.exp(-A[i][j])),3) 
	return A

def d_sigmoid(y):
	return np.round(y*(1-y),3)

class NeuralNetwork:
	"""
		A Fully connected Multilayer Neural Network class (based on numpy arrays).
		Hidden activation => sigmoid
		Output activation => sigmoid
		Loss => Cross-Entropy
	"""
	W = None ## List of weight matrices
	b = None ## List of bias vectors
	a = None ## List of pre-activation functions
	h = None ## List of activation functions
	y_cap = None ## Output value/vector

	def __init__(self, n_features, n_Output, n_layers = 2, lr = 0.01):
		self.input_neurons = n_features
		self.hidden_neurons = n_features
		self.output_neurons = n_Output
		self.layers = n_layers

		## Define the weight matrices and biases.
		self.W = [[] for i in range(self.layers+1)]
		self.b = [[] for i in range(self.layers+1)]
		self.W[0] = None ## Just for Notational clarity
		self.b[0] = None ## Just for Notational clarity
		for i in range(1,self.layers+1):
			if i < self.layers:
				self.W[i] = np.random.randn(self.hidden_neurons,self.input_neurons)
				self.b[i] = np.random.randn(self.hidden_neurons,1)
			elif i == self.layers:
				self.W[i] = np.random.randn(self.output_neurons,self.hidden_neurons)
				self.b[i] = np.random.randn(self.output_neurons,1)

		## Predefining activation and pre-activation Lists.
		self.a = [[] for i in range(self.layers+1)]
		self.h = [[] for i in range(self.layers)]

		## Learning rate
		self.lr = lr

	def feedforward(self,Input_sample):
		"""
			Feedforward process
		"""
		Input = np.reshape(Input_sample,(-1,1))

		self.a[0] = None
		self.h[0] = Input
		
		for k in range(1,self.layers):
			self.a[k] = self.b[k] + np.dot(self.W[k],self.h[k-1])
			self.h[k] = sigmoid(self.a[k])
		self.a[self.layers] = self.b[self.layers] + np.dot(self.W[self.layers],self.h[self.layers-1])
		self.y_cap = sigmoid(self.a[self.layers])

		return self.y_cap
	
	def backprop(self,output_List):
		## Lists for gradients.
		grad_W = [[] for i in range(self.layers+1)]
		grad_b = [[] for i in range(self.layers+1)]
		grad_a = [[] for i in range(self.layers+1)]
		grad_h = [[] for i in range(self.layers)]
		grad_W[0] = None
		grad_b[0] = None
		grad_a[0] = None
		## Compute output gradients.
		l = output_List.index(1)
		y_L = np.zeros((self.output_neurons,1))
		y_L[l] = -1*(1/(self.y_cap[l]+0.001))
		d_a = d_sigmoid(self.a[self.layers])
		grad_out = y_L*d_a

		grad_a[self.layers] = grad_out

		for k in range(self.layers,0,-1):
			#Compute grads w.r.t parameters.
			grad_W[k] = np.dot(grad_a[k],self.h[k-1].T)
			grad_b[k] = grad_a[k]

			#Compute grads w.r.t layers below.
			grad_h[k-1] = np.dot(self.W[k].T,grad_a[k])
			if k != 1:
				d_a1 = d_sigmoid(self.a[k-1])
				grad_a[k-1] = grad_h[k-1]*d_a1

		return grad_W,grad_b

	def Train(self,Input_sample,Output_List):
		max_iter = 2000
		for i in range(max_iter):
			self.feedforward(Input_sample)
			grad_W,grad_b = self.backprop(Output_List)
			for i in range(1,self.layers+1):
				#weight update using SGD step
				self.W[i] = self.W[i] - self.lr*grad_W[i]
				self.b[i] = self.b[i] - self.lr*grad_b[i]
		#Resetting pre-activation and activation layers to default for easy initialization.
		self.a = [[] for i in range(self.layers+1)]
		self.h = [[] for i in range(self.layers)]
		#Set y_cap to default.
		self.y_cap = None
