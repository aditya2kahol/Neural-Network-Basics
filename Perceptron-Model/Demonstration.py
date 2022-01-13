"""
	A quick demonstration of the perceptron model.
	Author: Aditya Kahol
"""

from random import uniform as unif
from Perceptron import Perceptron
from Points import Points
from random import seed

def Construct_training_data(size):
	I_values = []
	O_values = []
	for i in range(size):
		x = round(unif(-10,10),3)
		y = round(unif(-10,10),3)
		P = Points(x,y)
		I_values.append(P.return_points())
		O_values.append(P.return_label())

	return [I_values,O_values]

seed(10)
size = 150

My_data = Construct_training_data(size)

X = My_data[0]
Y = My_data[1]

P = Perceptron(size = 2)

#Train the perceptron Model
for i in range(100):
	inputs = X[i]
	target = Y[i]
	P.Train(inputs,target)

#Check Predictions.
for i in range(100,150):
	b = P.guess(X[i])
	y = Y[i]
	print(f"{i-99}) Guess for {X[i]} = {b} | {b == y}")