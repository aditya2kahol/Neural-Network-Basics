"""
	Points: Training data

	Labelling: Classes = 2: 
				if x > y: Class +1
				else	: Class -1 
"""

class Points:
	x = None
	y = None
	label = None

	def __init__(self,x,y):
		self.x = x
		self.y = y
		if x > y:
			self.label = 1
		else:
			self.label = -1

	def return_points(self):
		return [self.x,self.y]

	def return_label(self):
		return self.label