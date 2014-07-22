#going to test out the cython:
import numpy as np

import flowField
reload(flowField)

import step as dog
reload(dog)

import matplotlib.pyplot as plt
from pylab import *

#python -m cProfile -o profile/profile plumeSim.py













def main():
	#x = np.ndarray([5,1])
	#y = np.ndarray([5,1])
	#for a, b in zip(x,y):
	#	a = 12.0
	#	b = 22.0
	x = np.zeros([500, 1], dtype=float)
	y = np.zeros([500, 1], dtype=float)
	x = np.asarray([12.0 for xi in x])
	y = np.asarray([22.0 for yi in y])
		
	#x = np.array([12.0,12.0,12.0,12.0,12.0])
	#y = np.array([22.0,22.0,22.0,22.0,22.0])

	flow = flowField.flowField('simple')
	vx, vy = flow.getVectorSoA(x, y)

	#print vy, vx #this is correct
	vx = np.asarray(vx)
	vy = np.asarray(vy)

	##print vx[-1]
	#print vy[-1]
	for i in xrange(150):
		print i
		x, y = dog.kinzelbach1990SoA(x, y, vx, vy)

	#print "in main"
	#print x[0]
	#print y[-1]
	plt.scatter(x,y)
	plt.axis([0,20,0,30])
	show()

if __name__ == '__main__':
	main() 


