#flow field

import scipy.io as sio
from numpy import *
from pylab import *



mat_contents 	= sio.loadmat('constants.mat')
VX 				= mat_contents['V_x_matrix']
VY				= mat_contents['V_y_matrix']
xstep				= mat_contents['x']
ystep				= mat_contents['y']



class flowField():
	def __init__(self, which = 'mit'):
		#this is a holder flow field that we can use for testing
		#will soon be replaced by something real.

		if which == 'mit':
			self.vx = VX
			self.vy = VY
			self.x = xstep
			self.y = ystep
		if which == 'simple':
			x = arange(0, 30.2, .2)
			y = arange(0, 20.2, .2)
			self.x, self.y = meshgrid(y,x)
			self.vx = -1 *ones(shape(self.x))
			self.vy = zeros(shape(self.y))


	def getVectorSoA(self, x, y):
		#print "flowField"
		if (size(x) != size(y)):
			print "NOOOO!!!!!! Look at flowField.getVectorSoA"




		sizX = size(x)
		#print("Max x: "+str(max(x))+" Min x: "+str(min(x))+ \
		#	"\nMax y: "+str(max(y))+ " Min y: " +str(min(y)) ) 


		xV = [(xi/30)*151 for xi in x]
		yV = [(yi/20)*101 for yi in y]
		xA = [self.vx[int(xi), int(yi)] for xi, yi in zip(xV,yV)]
		yA = [self.vy[int(xi), int(yi)] for xi, yi in zip(xV,yV)]

		return xA, yA

	def getVal(self, x, y):
		#this should be singular values
		xV = (x/30)*151
		yV = (y/20)*101
		#print "xV: %s yV: %s" %(xV, yV)

		vx = self.vx[xV,yV]
		vy = self.vy[xV,yV]


		#vx = (vx, 0)[math.isnan(vx)]
		#vy = (vy, 0)[math.isnan(vy)]
		return vx, vy



	def getVector(self,x,y):
		print "flowField"
		x = (x/30)*151
		y = (y/20)*101
		
		
		#print x, y
		try:	
			a = self.vx[x, y]
		except IndexError:
			print x, y
			print "Not a valid value for getVector: (%s, %s)"%(x,y)
			raise
			

		try:
			b = self.vy[x, y]
		except IndexError:
			print x, y
			print "Not a valid value for getVector: (%s, %s)"%(x,y)
			raise
		

		#just in case	
		a = (a, 0)[math.isnan(a)]
		b = (b, 0)[math.isnan(b)]

		return a,b


