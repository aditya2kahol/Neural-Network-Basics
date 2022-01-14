"""
	Perceptron model (A single layer single neuron model)
	Author: Aditya Kahol
	Reference: https://www.youtube.com/watch?v=ntKn5TPHHAk&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=3&t=217s
"""

from random import uniform as unif

class Perceptron:
	
	weights = None
	
	#Constructor, which initializes weights at random.
	def __init__(self,size):
		self.size = size + 1 # one extra weight vector for the bias term 
		self.weights = []
		for i in range(size):
			self.weights.append(round(unif(-1,1),4))

	#My Activation function (here, it is signum function)
	@staticmethod
	def Sign(x):
		if x >= 0:
			return 1
		else:
			return -1

	#My Output function, which returns the guessed result.
	def guess(self,inputs):
		Sum = 0
		for i in range(self.size):
			Sum += self.weights[i]*inputs[i]
		self.Output = self.Sign(Sum)
		return self.Output

	#Train Function
	def Train(self,inputs,target,lr = 0.1):
		#Function to train the perceptron.
		#Note: Actual error used is mse: i.e (target - my_guess)^2, followed by the gradient descent algorithm. 
		my_guess = self.guess(inputs)
		error = target - my_guess
		for i in range(self.size):
			#weight update step
			self.weights[i] += lr*error*inputs[i]
