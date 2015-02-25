"""

Nathaniel Saul 2014
UROP plume simulation
Field Robotics Laboratory
University of Hawaii

This module holds the class, inherited from our main plume class, 
that handles all of the functionality used in characterizing and
analyzing the generated plume.  It has functions that handle
various kinds of plot generations and functions that support 
various specific experiments that are in the plumeSim function

"""












'''

One measure is intermittency ( see c.d. Jones 'On the structure of instanteous plumes in the atmosphere)
We could see an intermittency of upwards of 80%



'''
import sys
import os



import scipy.stats as ss
from scipy.interpolate import UnivariateSpline
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib 
from numpy import *
from pylab import *


import plumeClass
reload(plumeClass)



class plumeExpC(plumeClass.plume):
	def __init__(self, parameters):
		plumeClass.plume.__init__(self, parameters)
		self.gather = []
		for i in range(3):
			self.gather.append(list())


	def gatherPointsSoA(self):
		line = [ self.param.yi - 5, self.param.yi - 10, self.param.yi - 15 ]
		margin = 0.5
		yp, xp = self.getPointsSoA()
		for each, x in zip(yp, xp):
			for l, gat in zip(line, self.gather): 
				if (l-margin) < each <(l+margin):
					gat.append(x)


	def plotPDF(self):	
		print "plot the PDF stuffs"
		figure(1)
		print shape(self.gather)
		for each in self.gather:
			print len(each)

		smoothness = 75
		kde = []; distSpace = []; p = []; txt = []
		for i in range(3):
			kde.append( list() )
			distSpace.append( list() )
			p.append( list() )
			txt.append( list() )
	
		
		lbl = [5,10,15]
		distxx = [0.25, 0.50, .75]
		for kd, dS, gat, pl, lb in \
			zip(kde, distSpace,self.gather, p, lbl):
			kd = gaussian_kde(gat)
			dS = linspace( min(gat), max(gat), smoothness)
			pl = plt.plot(dS, kd(dS), label = "%s units from source" %lb)

		
	

		title("Probability density function of plume down stream from source")

		mean = 12; variance = 1
		sigma = np.sqrt(variance)
		x = linspace(9,15,100)
		plt.plot(x,mlab.normpdf(x,mean,sigma), label = "normal distribution")
	
		for lb, t, gat, dis in zip(lbl, txt, self.gather, distxx):
			t = ("%s units from source:\nskew: %4.4f\nvariance: %4.4f" \
				%(lb, ss.skew(gat) , ss.tvar(gat)))

			xloc = xlim()[0]+0.15*diff(xlim())
			yloc = ylim()[0]+dis*diff(ylim())
			text(xloc, yloc, t)
		plt.legend()
		plt.show()		
					
		

	def densityPlotHex(self): 
		"""
			Plot the density hex bin plot of the entire plume.
		"""
		
		xp, yp = self.getPointsSoA()

		figure()
		hexbin(yp,xp, bins = 'log', cmap=plt.cm.hsv)
		axis('equal')
		#		axis([0, 20, 0, 30])
		cb = colorbar()
		show()

	def horizontalDistribution(self):
		points = []
		for each in self.puffs:
			points.append( each.loc[1] )
		std = findSTD(points)
		return std





	def plotPDFandData(self):

		smoothness = 75
		kde7 = gaussian_kde( self.seventeens )
		dist_space7 = np.linspace( min(self.seventeens ), max(self.seventeens ), smoothness )
		p7 = plt.plot( dist_space7, kde7(dist_space7), label = "10 units from source" )

		pReal = plt.hist(self.seventeens, 200)

		txt7 = ("seventeens:\nskew: %4.4f\nkurtosis: %4.4f\nvariance: %4.4f" %( 
			ss.skew(self.seventeens), ss.kurtosis(self.seventeens), ss.tvar(self.seventeens) ))
		xloc = xlim()[0]+0.15*np.diff(xlim())

		yloc = ylim()[0]+0.50*diff(ylim())
		text(xloc, yloc, txt7)

		plt.legend()
		plt.show()





'''
	def gatherPoints(self):
		"""
		find the time averaged probability density function at a certain level
			choose 5, 10, and 15 points down from the source location
			find and add to the list all of the points that are within 
			a small margin of this line
		"""
		
	
		line1 = self.source[0] - 5
		line2 = self.source[0] - 10
		line3 = self.source[0] - 15
			
		
		margin = .5
		
		for each in self.puffs:
			if (each.loc[0] - line1) < margin and (each.loc[0] - line1) > -margin:
				self.twentytwos.append(each.loc[1])

			elif (each.loc[0] - line2) < margin and (each.loc[0] - line2) > -margin:
				self.seventeens.append(each.loc[1])
			elif (each.loc[0] - line3) < margin and (each.loc[0] - line3) > -margin:
				self.tens.append(each.loc[1])
		

'''



"""


	def distribution(self):
		
			Older functions


		
		print 'begin distribution'
		c = zeros((151,101))
		X = arange(0, 30.2, .2)
		Y = arange(0, 20.2, .2)
		X, Y = meshgrid(Y, X)
		total = 0

		#this goes through a grid and finds the concentration at each vertice
		for i in xrange(151):
			for j in xrange(101):
				#print i,j
				c[i,j] = int( self.concentration(i/5,j/5, .1) )
				total += c[i,j]	

		cst = c.astype(str)

		savetxt('concentrationMatrix.txt', cst, delimiter="", fmt="%s")

		print "found in concentration: %s\ntotal puffs: %s" \
				%(total, shape(self.puffs))
	



def conOnCenter(self):
		#this can only be used when using 'simple' vector field
		print 'begin conOnCenter'
		print self.source[1]

		#for 1 - 151, we are going to take all these steps

		#lets say 5 steps per 151:  

		gridSize = 151
		steps = 5
		tSteps = gridSize*steps

		c = []
		for i in xrange(151):
			c.append( self.concentration(i/5, self.source[1], .1) )

		index = arange(0,30.2,.2)
		figure(3)
		bar(index, c, .15,
                 color='b',)
		show()







"""
