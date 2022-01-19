"""
	Matrix class in python bulit from scratch.
	Author: Aditya Kahol
"""

from random import uniform as unif

class Matrix:
	
	Mat = None
	
	def __init__(self,r=2,c=2): # A 2x2 zero matrix will be made by default, if no arguments are passed.
		self.rows = r
		self.cols = c
		self.shape = (r,c)
		self.Mat = []
		for i in range(self.rows):
			self.Mat.append([])
			for j in range(self.cols):
				self.Mat[i].append(0)

	def __str__(self):
		"""
			Used to display the matrix element.
			Just like in numpy.
		"""
		for i in range(self.rows):
			if i == 0:
				if i == self.rows-1:
					print(f"[{self.Mat[i]}]")
				else:
					print(f"[{self.Mat[i]}")
			elif i == self.rows-1:
				print(f" {self.Mat[i]}]")
			else:
				print("",self.Mat[i])
		return ""

	@classmethod
	def randomize(cls,r=2,c=2,k = 1):
		#Function to generate a random matrix with element values ranging from values [-k,k]
		M = cls(r,c)
		for i in range(M.rows):
			for j in range(M.cols):
				M.Mat[i][j] += round(unif(-1,1)*k,3)
		return M

	@classmethod
	def randint(cls,r=2,c=2,k = 10):
		#Function to generate a Matrix object with random integer values from [-k,k]
		M = cls(r,c)
		for i in range(M.rows):
			for j in range(M.cols):
				M.Mat[i][j] += round(unif(-1,1)*k)
		return M
	
	@classmethod
	def toMatrix(cls,L):
		"""
			Constructor function to create a Matrix object which takes elements from a list.
			Input: a list L
			Output: a Matrix object.
		"""
		r = len(L)
		try:
			c = len(L[0])
		except:
			c = 1
		K = cls(r,c)
		if c != 1:
			for i in range(r):
				for j in range(c):
					K.Mat[i][j] = L[i][j]
			return K
		else:
			for i in range(r):
				for j in range(c):
					K.Mat[i][j] = L[i]
			return K

	def map(self,func,*args):
		if len(args) == 0:
			for i in range(self.rows):
				for j in range(self.cols):
					self.Mat[i][j] = func(self.Mat[i][j])
		else:
			for i in range(self.rows):
				for j in range(self.cols):
					self.Mat[i][j] = func(self.Mat[i][j],args[0])

	@staticmethod
	def map(A,func,*args):
		K = Matrix(A.rows,A.cols)
		if len(args) == 0:
			for i in range(A.rows):
				for j in range(A.cols):
					K.Mat[i][j] = func(A.Mat[i][j])
		else:
			for i in range(A.rows):
				for j in range(A.cols):
					K.Mat[i][j] = func(A.Mat[i][j],args[0])
		return K

	def transpose(self):
		K = Matrix(self.cols,self.rows)
		for i in range(self.rows):
			for j in range(self.cols):
				K.Mat[j][i] = self.Mat[i][j]
		return K

	def Hadamard(self,M):
		"""
			Function for elementwise multiplication for matrices of same ordered matrices.
		"""
		H = Matrix(self.rows,self.cols)
		for i in range(self.rows):
			for j in range(self.cols):
				H.Mat[i][j] = self.Mat[i][j]*M.Mat[i][j]
		return H

	def MoreThan(self,M):
		for i in range(self.rows):
			for j in range(self.cols):
				if self.Mat[i][j] < M.Mat[i][j]:
					return False
		return True
		
	## Defining some important dunder methods for this class.
	def __add__(self,A):
		if not isinstance(A,Matrix):
			K = Matrix(self.rows,self.cols)
			for i in range(self.rows):
				for j in range(self.cols):
					K.Mat[i][j] = round(self.Mat[i][j] + A,3)
			return K
		
		if self.rows == A.rows and self.cols == A.cols:
			K = Matrix(self.rows,self.cols)
			for i in range(self.rows):
				for j in range(self.cols):
					K.Mat[i][j] = round(self.Mat[i][j] + A.Mat[i][j],3)
			return K
		else:
			print("Matrices are not conformable, cannot add")
			return None

	def __sub__(self,A):
		if not isinstance(A,Matrix):
			K = Matrix(self.rows,self.cols)
			for i in range(self.rows):
				for j in range(self.cols):
					K.Mat[i][j] = round(self.Mat[i][j] - A,3)
			return K
		
		if self.rows == A.rows and self.cols == A.cols:
			K = Matrix(self.rows,self.cols)
			for i in range(self.rows):
				for j in range(self.cols):
					K.Mat[i][j] = round(self.Mat[i][j] - A.Mat[i][j],3)
			return K
		else:
			print("Matrices are not conformable, cannot subtract")
			return None
	
	def __mul__(self,A):
		#Scalar Product
		if isinstance(A,Matrix) == False:
			K = Matrix(self.rows,self.cols)
			for i in range(self.rows):
				for j in range(self.cols):
					try:
						K.Mat[i][j] = round(self.Mat[i][j] * A,3)
					except:
						print("Scalar multiplication is possible only with numbers")
						return None
			return K
		else:
			#Matrix product
			if self.cols == A.rows:
				K = Matrix(self.rows,A.cols)
				for i in range(self.rows):
					for j in range(A.cols):
						Sum = 0
						for k in range(self.cols):
							Sum += self.Mat[i][k]*A.Mat[k][j]
						K.Mat[i][j] = Sum
				return K
			else:
				print("Matrices are not conformable, cannot multiply")
				return None
