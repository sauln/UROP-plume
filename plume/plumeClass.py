#we need to start over

#we are going to have the tight class that hangles the plume

import os
import gc
import psutil
import sys
import copy
import time



import math
from numpy import *
import numpy as np
from pylab import *
import cPickle



import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#import auxiliary
#reload(auxiliary)

import flowField
reload(flowField)

import step
reload(step)

eps = np.finfo(float).eps




class plumeEtAl():
	def __init__(self, param = None,  load = False, fileName = "plumeHist" ):
		self.param = param
		self.plumeHist = []
		self.fileName = fileName
		#self.fileName = fileName


		print load
		if load: #if we need to create the 
			print "Load sim from file, do not generate."
			self.loadData(fileName)

	def add(self, mostRecent, T):
		#in here, we want to save off every second...
		#and clear the plumeEtAl all except the last entry, and then continue 
		#from there with full memory.  
		print shape(self.plumeHist)

		print "T: %s" %T
	
		if T%(1/self.param.dt) == 0 and T != 0:
			print "Save off data and start with clean slate."
			T = T/(1/self.param.dt) 
			
			self.saveData(T)


			print "MEMORY USAGE BEFORE: %s" %self.memory_usage_psutil()
			
			del(self.plumeHist)
			#reset(self.plumeHist)
			gc.collect()
			self.plumeHist = []
			
			print "MEMORY USAGE AFTER: %s" %self.memory_usage_psutil()


		self.plumeHist.append(copy.copy(mostRecent))


	def memory_usage_psutil(self):
		# return the memory usage in MB
		import psutil
		process = psutil.Process(os.getpid())
		mem = process.get_memory_info()[0] / float(2 ** 20)
		return mem
		



	def printAll(self):
		for each in self.plumeHist:
			print each.xs[1]
		for each in self.plumeHist:
			print each.ys[1]


	def loadData(self, fileName, T = 1):
		""" URGENT TODO """
		""" 
			This needs to automatically load 
			the next file when it gets to the
			end.  Right now, it only runs for
			the first one!!
		"""

		print "use cPickle to load %s" % fileName
		start = time.clock()
		print "MEMORY USAGE BEFORE DELETION: %s" %self.memory_usage_psutil()
		if hasattr(self, "plumeHist"):
			print "delete stuff of size: %s" % sys.getsizeof(self.plumeHist)
			del self.plumeHist 
			gc.collect()
		print "MEMORY USAGE AFTER DELETION: %s" %self.memory_usage_psutil()

		filename = "data/%s_%s.p" %(fileName, T)
		print "Going to load data from %s" % filename
		f = open(filename,'rb')
		tmp_dict = cPickle.load(f)
		f.close()  
		'''        
		if hasattr(self, "__dict__"):
			del(self.__dict__)
			#reset(self.plumeHist)
			gc.collect()
		'''

		


		self.__dict__.update(tmp_dict) 
		elapsed = (time.clock() - start)
		print "cPickle took %s seconds long" % elapsed
		
			

	def saveData(self, t = 0):

		filename = "data/%s_%s.p" %(self.fileName, int(t))
		#fileName1 = "data/%s_%s_%s.p" %(t, self.param.dt, self.param.den)
		f = open(filename,'wb')
		cPickle.dump(self.__dict__,f,2)
		f.close()
		print "saving file to %s" %filename

	def makeMovie(self, fig):
		print "begin making movie"

		for each in self.plumeHist:
			#plt.clf()
			#ax = fig.add_subplot(111, aspect='equal')
			#plt.axis([0,20,0,30])
			
			if fig.sc !=None:
				fig.sc.remove()
	
			fig.sc = plt.scatter(each.ys[::10],each.xs[::10])
			
			sc = scatter(12, 26, s = 100,c = 'r', marker='o', zorder = 1)

			Q = quiver( self.flow.x[::6,::6], self.flow.y[::6,::6], \
				self.flow.vy[::6, ::6], self.flow.vx[::6, ::6], 
				color = 'r', units='x',pivot=( 'tail' ),
		        linewidths=(1,), edgecolors=('r'), headaxislength=2 )

			plt.title('Visualization of sim %s\nPlume Update: %s seconds'%\
				(self.fileName, self.plumeHist.index(each)*self.param.dt))
			plt.draw()

		print "I'm done making a movie now"


class puffSoA():
	#using a structure of arrays instead of an array 
	#of structures allows us to do calculations in a
	#very pythonic way.  Is it actually faster though?



	
	def __init__(self):
		self.xs = []
		self.ys = []

	#This needs to be rewritten as numpy arrays...


	def gradientDivergence(self, y, x, vx, vy, r, norm):
		#print "VY: %s VX: %s\nY: %s X: %s" %(vy, vx, y, x)



		yU = self.concentration(y+r, x, r)	* norm
		yD = self.concentration(y-r, x, r)	* norm
		xU = self.concentration(y, x+r, r)  * norm
		xD = self.concentration(y, x-r, r)  * norm
		c = self.concentration(y,x,r)       * norm

		#print "yU: %s yD: %s\nxU: %s xD: %s\nc: %s "%(yU, yD, xU, xD, c)
		#print "vx: %s vy: %s"%(vx,vy)
		DU_dy0 = (vy>=0)*(yU-c)/(r)+(vy<0)*(c-yD)/(r)
	  	DU_dx0 = (vx>=0)*(xU-c)/(r)+(vx<0)*(c-xD)/(r)

		D2U0 = (yU+yD-2*c)/r**2+(xU+xD-2*c)/r**2
		return DU_dx0, DU_dy0, D2U0




	def concentration(self, y, x, r):

		start = time.clock()
		xT =  [(xs<x+r) & (x-r<xs) for xs in self.xs]
		yT =  [(ys<y+r) & (y-r<ys) for ys in self.ys]
		both = [tx and ty for tx, ty in zip(xT, yT)]
		tot = len( [i for i,j in enumerate(both) if j == True] )
		elapsed = (time.clock() - start)
		#print "list way took %s seconds long\n	answer: %s" % (elapsed,tot)

		return tot


	def addPuffs(self, locX, locY, count):
		for x in xrange(count):
			self.xs.append(locX)
			self.ys.append(locY)



class plume():
	def __init__(self, parameters, fileName, gen = True):
		self.plumeEtAl = plumeEtAl(parameters, gen, fileName)
		self.param = parameters
		self.flow = flowField.flowField(self.param.flow)
		self.plumeEtAl.flow = self.flow
		self.puffSoA = puffSoA()
		self.puffQueue = 0
	
	def tickSoA(self, T):
		#this racks up puff debt, when it reaches a whole puff, it makes it
		#or, every time, it makes all the puffs
		self.puffQueue += self.param.ratio
		if self.puffQueue >= 1:
			self.puffSoA.addPuffs(self.param.yi, self.param.xi, \
				int(floor(self.puffQueue)))
			self.puffQueue -= int(floor(self.puffQueue))
		self.movePuffsSoA()



		self.plumeEtAl.add(self.puffSoA, T)




	def getPointsSoA(self):
		return self.puffSoA.xs, self.puffSoA.ys
		
	def movePuffsSoA(self):
		#self.kinzelbach1990SoA()
		self.doCythonKinzelbach()
		#self.doSameKinzelbach()


	def doSameKinzelbach(self):
		#this is supposed to be as similar to step.kinzelbach as possible
		#for verification purposes
		self.puffSoA.xs, self.puffSoA.ys = \
			self.KinzelbachSame(self.flow, self.puffSoA.xs, self.puffSoA.ys)

	def doCythonKinzelbach(self):
		#print self.puffSoA.xs
		#print self.puffSoA.ys
		#print "get flow field"
		vx, vy = self.flow.getVectorSoA(self.puffSoA.xs, self.puffSoA.ys)

		#print vy, vx #this is correct
		vx = np.asarray(vx)
		vy = np.asarray(vy)

		self.puffSoA.xs, self.puffSoA.ys = step.kinzelbach1990SoA( \
			np.asarray(self.puffSoA.xs), \
			np.asarray(self.puffSoA.ys), vx, vy  )



		self.puffSoA.xs = self.puffSoA.xs.tolist()
		self.puffSoA.ys = self.puffSoA.ys.tolist()


	def createPuffSoA(self, count):
		#this is deprecated, use the puffSoA class method instead
		self.puffSoA.add(self.param.xi, self.param.yi, count)





	def saveTick(self):
		#no clue how to do this...
		pass



	def plotPlume(self):
		flow = self.flow
		close('all')
		print "plot plume"
		
	
		figure(2)
		Q = quiver( flow.x[::6,::6], flow.y[::6,::6], \
				flow.vy[::6, ::6], flow.vx[::6, ::6], 
				color = 'r', units='x',pivot=( 'tail' ),
		        linewidths=(2,), edgecolors=('k'), headaxislength=5 )
	
		#plot the puffs
		xp, yp = self.getPointsSoA()
		s = scatter(yp,xp)

		sc = scatter(self.param.xi, self.param.yi, s = 100,c = 'r', marker='o')
		xl = xlabel('$x$'); yl = ylabel('$y$'); ax = axis('image')
	
		title("Density: %s\nTime: %s"% (self.param.den, self.param.T))
		show()





	"""
	def kinzelbach1990SoA(self):
		''' 
			Equation described in Kinzelback1990
			adapted to incorporate a structure of arrays instead 
			of an array of structures
		'''
		D = 0.5 #diffusion coefficient
		
		numX = shape(self.puffSoA.xs)[0]
		numY = shape(self.puffSoA.ys)[0]
		if (numX != numY):
			for x in xrange(10):
				print "NOT THE SAME! SOMETHING IS WRONG!"

		#considering this cycles through (i in xrange(numX)) so many times
		#we can probably consolidate them

		#to debug, we are going to use 1 instead of a random number
		alphaXSoA = [0 for i in xrange(numX)]
		alphaYSoA = [0 for i in xrange(numX)]


		#alphaXSoA = [np.random.normal(0,1) for i in xrange(numX)]
		#alphaYSoA = [np.random.normal(0,1) for i in xrange(numX)]

		vx, vy = self.flow.getVectorSoA(self.puffSoA.xs, self.puffSoA.ys)

		vxFix = [(vx[i] == 0)*1.0 + (vx[i] != 0)*vx[i] for i in xrange(numX)]
		vyFix = [(vy[i] == 0)*1.0 + (vy[i] != 0)*vy[i] for i in xrange(numX)]

		advX = [vx[i]*self.param.dt for i in xrange(numX)]
		advY = [vy[i]*self.param.dt for i in xrange(numX)]

		difX = [alphaXSoA[i] * ((abs(2 * D * vxFix[i] * \
			self.param.dt)) ** .5  ) for i in xrange(numX)]
		difY = [alphaYSoA[i] * ((abs(2 * D * vyFix[i] * \
			self.param.dt)) ** .5  ) for i in xrange(numX)]

		self.puffSoA.xs = [self.puffSoA.xs[i] + advX[i] + difX[i] \
			for i in xrange(numX)]
		self.puffSoA.ys = [self.puffSoA.ys[i] + advY[i] + difY[i] \
			for i in xrange(numX)]
	



		#remove all puffs below 0 and above 20 or 30

		for x in self.puffSoA.xs:
			if x<=0 or x>=30:
				i = self.puffSoA.xs.index(x)
				self.puffSoA.xs.pop(i)
				self.puffSoA.ys.pop(i)
				print "Erroneous value!! %s:"%x

		for y in self.puffSoA.ys:
			if y<=0 or y>= 20:
				i = self.puffSoA.ys.index(y)
				self.puffSoA.xs.pop(i)
				self.puffSoA.ys.pop(i)
				print "Erroneous value!! %s:"%y
	
	"""
	"""

	def KinzelbachSame(self, flow, xs, ys):
		''' 
			Equation described in Kinzelback1990
			adapted to incorporate a structure of arrays instead 
			of an array of structures
		'''
		D = 0.5 #diffusion coefficient
		
		numX = shape(xs)[0]
		numY = shape(ys)[0]
		if (numX != numY):
			for x in xrange(10):
				print "NOT THE SAME! SOMETHING IS WRONG!"

		#considering this cycles through (i in xrange(numX)) so many times
		#we can probably consolidate them

		#to debug, we are going to use 1 instead of a random number
		alphaXSoA = [0 for i in xrange(numX)]
		alphaYSoA = [0 for i in xrange(numX)]


		#alphaXSoA = [np.random.normal(0,1) for i in xrange(numX)]
		#alphaYSoA = [np.random.normal(0,1) for i in xrange(numX)]

		vx, vy = flow.getVectorSoA(xs, ys)

		vxFix = [(vx[i] == 0)*1.0 + (vx[i] != 0)*vx[i] for i in xrange(numX)]
		vyFix = [(vy[i] == 0)*1.0 + (vy[i] != 0)*vy[i] for i in xrange(numX)]

		advX = [vx[i]*self.param.dt for i in xrange(numX)]
		advY = [vy[i]*self.param.dt for i in xrange(numX)]

		difX = [alphaXSoA[i] * ((abs(2 * D * vxFix[i] * \
			self.param.dt)) ** .5  ) for i in xrange(numX)]
		difY = [alphaYSoA[i] * ((abs(2 * D * vyFix[i] * \
			self.param.dt)) ** .5  ) for i in xrange(numX)]

		xs = [xs[i] + advX[i] + difX[i] \
			for i in xrange(numX)]
		ys = [ys[i] + advY[i] + difY[i] \
			for i in xrange(numX)]
	



		#remove all puffs below 0 and above 20 or 30

		for x in xs:
			if x<0 or x>30:
				i = xs.index(x)
				xs.pop(i)
				ys.pop(i)
				print "Erroneous value!! %s:"%x

		for y in ys:
			if y<0 or y> 20:
				i = ys.index(y)
				xs.pop(i)
				ys.pop(i)
				print "Erroneous value!! %s:"%y
		return xs, ys
	
	"""

	'''
	def movePuffs(self):
		#self.kitanidisNaive()
		self.kinzelbach1990()
		#self.originalMove()
	
	'''	

	"""
	def kitanidisNaive(self):

		''' This is the 'simplistic approach' presented by Kitanidis
		'''
		#x(t+dt) = x(t) + u( x(t) ) dt + [2 D x(t) dt] ** .5 * e
		#u is the advective velocity
		#e is a normal random variable with zero mean and unit variance
		#D is the dispersion coefficient - this is a function

		D = .2 #this is the diffusion coefficient that Shuai says in his paper
		#dt= 0.12
		#print "got here"
		for each in self.puffs:
			vy, vx = self.flow.getVector(each.loc[0], each.loc[1]) 

			dif = np.random.normal(0, 1)
			adv = np.random.normal(0, 1)

			#print each.loc[0]
			
			yd = vy*self.settings.dt + ((2 * D * self.settings.dt) ** .5 ) * dif
			each.loc[0] = each.loc[0] + yd 

			xd = vx*self.settings.dt + ((2 * D * self.settings.dt) ** .5 ) * adv
			each.loc[1] = each.loc[1] + xd


			if(each.loc[0] >=30 or each.loc[0] <= 0):
				self.puffs.remove(each)
				self.removed +=1
			elif(each.loc[1] >= 20 or each.loc[1] <=0):
				self.puffs.remove(each)
				self.removed +=1

	def kinzelbach1990(self):
		''' 
			Equation described in 
		'''
		Z = 0.5 #diffusion coefficient

		for each in self.puffs:
			#random variables in x and y direction and velocity vector
			alphaX = np.random.normal(0, 1)
			alphaY = np.random.normal(0, 1)
			vy, vx = self.flow.getVector(each.loc[0], each.loc[1]) 
			
			#this will correct for when one of the velocity components is 0
			vxU = (vx == 0)*1.0 + (vx != 0)*vx
			vyU = (vy == 0)*1.0 + (vy != 0)*vy
			
			advX = vx*self.settings.dt
			difX = alphaX * (  (abs(2 * Z * vxU * self.settings.dt)) ** .5  )
			each.loc[1] = each.loc[1] + advX + difX

			advY = vy*self.settings.dt
			difY = alphaY * (  (abs(2 * Z * vyU * self.settings.dt )) ** .5  ) 
			each.loc[0] = each.loc[0] + advY + difY 

			if(each.loc[0] >=30 or each.loc[0] <= 0):
				self.puffs.remove(each)
				self.removed +=1
			elif(each.loc[1] >= 20 or each.loc[1] <=0):
				self.puffs.remove(each)
				self.removed +=1



	def originalMove(self):
		#This algorithm is the random walk I came up with
		A = array([[0, -1], [1,0]])
		for each in self.puffs:
			vx, vy = self.flow.getVector(each.loc[0], each.loc[1]) 
			
			loc = array([[vx], [vy]])
			normalV = dot(A, loc)

			dif = np.random.normal(0,  self.settings.spread)# * each.moves ) #increase with time
			adv = np.random.normal(1,  self.settings.spread/2)# * each.moves )
			
			each.loc[0] = each.loc[0] + self.settings.dt * \
				 (dif * float(normalV[0])/2 + adv * vx) 
			each.loc[1] = each.loc[1] + self.settings.dt * \
				 (dif * float(normalV[1]) /2+ adv * vy)

			if(each.loc[0] >=30 or each.loc[0] <= 0):
				self.puffs.remove(each)
				self.removed +=1
			elif(each.loc[1] >= 20 or each.loc[1] <=0):
				self.puffs.remove(each)
				self.removed +=1

	"""
	"""
	def concentration(self, x, y, radius):
		c = 0
		radius = abs(radius)
		for each in self.puffs:
			if x-radius <= each.loc[0] < x+radius and \
				y-radius <= each.loc[1] < y+radius:
				c = c+1
		return float(c)
	"""
	"""
	def createPuff(self):
		puf = puff(self.source)
		self.puffs.append( puf )
	"""

	"""
	def printPuffs(self):
		for each in self.puffs:
			print "puff: %4.2f %4.2f"%(each.loc[0], each.loc[1])

	def tick(self, t):
		#slightly flawed algorithm

		for i in xrange(self.settings.density):
			self.createPuff()
		self.kitanidisNaive(self.settings.dt)

	def tick2(self):
		#method with much more flexibility
		self.puffQueue += self.ratio
		if self.puffQueue >= 1:
			for x in xrange(int(floor(self.puffQueue))):
				self.createPuff()
				self.puffQueue -= 1
		self.movePuffs()


	def getPoints(self):
		xp = []; yp =[]
		for each in self.puffs:
			xp.append(each.loc[0])
			yp.append(each.loc[1])
		return xp, yp

	
		
	"""
	"""
	def plotPlumeX(self, x, y):
		flow = self.flow
		close('all')
		print "plot plume"
		xp = []; yp = []
	
		figure(2)
		Q = quiver( flow.x[::6,::6], flow.y[::6,::6], \
				flow.vy[::6, ::6], flow.vx[::6, ::6], 
				color = 'r', units='x',pivot=( 'tail' ),
		        linewidths=(2,), edgecolors=('k'), headaxislength=5 )
	
	
		#plot the puffs
		for each in self.puffs:
			xp.append(each.loc[0])
			yp.append(each.loc[1])
		s = scatter(yp,xp)

		sc = scatter(self.source[1], self.source[0], s = 100,c = 'r', marker='o')
		xl = xlabel('$x$'); yl = ylabel('$y$'); ax = axis('image')
	
		sc1 = scatter(x ,y, c = 'b', marker = '+')

		show()
		





	"""

'''GET RID OF THIS'''
"""
class simSettings():
	def __init__(self, density, fl, std, dt):
		self.density = density
		self.flow = fl
		self.spread = std
		self.dt = dt

"""
'''
def setupPlume( x, y):
	settings = simSettings(density, flow, std, dt)
	
	plum = plume(settings, y, x)
	plum.settings = settings
	return plum
'''
'''
class puff():#think of this as a c-struct
	def __init__(self, source):
		self.loc = []
		self.loc.append( float(source[0]) )
		self.loc.append( float(source[1]) )
		self.moves = 0 
'''


