#going to test out the cython:
import numpy as np

import flowField
reload(flowField)


#python -m cProfile -o profile/profile plumeSim.py

def kinzelbach1990SoA(flow, x, y):
	''' 
		Equation described in Kinzelback1990
		adapted to incorporate a structure of arrays instead 
		of an array of structures
	'''
	D = 0.5 #diffusion coefficient
	
	numX = np.shape(x)[0]
	numY = np.shape(y)[0]


	print "I DONT KNOW HOW TO USE SHAPE!!"
	print numX
	print numY

	if (numX != numY):
		for x in xrange(10):
			print "NOT THE SAME! SOMETHING IS WRONG!"

	#considering this cycles through (i in xrange(numX)) so many times
	#we can probably consolidate them
	alphaXSoA = [np.random.normal(0,1) for i in xrange(numX)]
	alphaYSoA = [np.random.normal(0,1) for i in xrange(numX)]

	vx, vy = flow.getVectorSoA(y, x)
	
	dt = 0.002

	vxFix = [(vx[i] == 0)*1.0 + (vx[i] != 0)*vx[i] for i in xrange(numX)]
	vyFix = [(vy[i] == 0)*1.0 + (vy[i] != 0)*vy[i] for i in xrange(numX)]

	advX = [vx[i]*dt for i in xrange(numX)]
	advY = [vy[i]*dt for i in xrange(numX)]

	difX = [alphaXSoA[i] * ((abs(2 * D * vxFix[i] * \
		dt)) ** .5  ) for i in xrange(numX)]
	difY = [alphaYSoA[i] * ((abs(2 * D * vyFix[i] * \
		dt)) ** .5  ) for i in xrange(numX)]

	xs = [x[i] + advX[i] + difX[i] \
		for i in xrange(numX)]
	ys = [y[i] + advY[i] + difY[i] \
		for i in xrange(numX)]
	

	return xs, ys


def main():
	x = [12,12,12,12,12]
	y = [22,22,22,22,22]

	flow = flowField.flowField('simple')

	

	for i in xrange(5000):
		x, y = kinzelbach1990SoA(flow, x, y)


	print x
	print y


if __name__ == '__main__':
	main() 


